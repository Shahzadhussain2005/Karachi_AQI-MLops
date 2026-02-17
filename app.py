"""
AQI Prediction Dashboard
========================
Data: MongoDB Atlas (fallback: data/cleaned_aqi_data_v2.csv)
Models: Scripts/models/
"""

import warnings
warnings.filterwarnings('ignore')
import os
import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Karachi AQI Prediction",
    page_icon="🌫️",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1f77b4;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #333; margin-top: 1rem;}
    .metric-card {background: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

def get_aqi_category(aqi):
    if aqi <= 50:   return 'Good', '#00e400'
    elif aqi <= 100: return 'Moderate', '#ffff00'
    elif aqi <= 150: return 'Unhealthy for Sensitive Groups', '#ff7e00'
    elif aqi <= 200: return 'Unhealthy', '#ff0000'
    elif aqi <= 300: return 'Very Unhealthy', '#8f3f97'
    else:            return 'Hazardous', '#7e0023'

# ============================================================================
# Load Data - MongoDB first, CSV fallback
# ============================================================================

@st.cache_data(ttl=3600)
def load_data():
    # --- Try MongoDB first ---
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        MONGO_URI = "mongodb+srv://nawababbas08_db_user:2Ja4OGlDdKfG6EvZ@cluster0.jnxn95g.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true"

        client = MongoClient(
            MONGO_URI,
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        client.admin.command('ping')

        db  = client['aqi_feature_store']
        df  = pd.DataFrame(list(db['aqi_features'].find({}, {"_id": 0})))
        client.close()

        if len(df) > 0:
            time_col = 'time' if 'time' in df.columns else 'timestamp'
            df['time'] = pd.to_datetime(df[time_col])
            df = df.sort_values('time').reset_index(drop=True)
            return df, '🟢 MongoDB Atlas'

    except Exception:
        pass  # fall through to CSV

    # --- CSV fallback ---
    csv_paths = [
        'data/cleaned_aqi_data_v2.csv',
        'data/cleaned_aqi_data_v3.csv',
        'data/historical_aqi.csv',
    ]
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            time_col = 'time' if 'time' in df.columns else 'timestamp'
            df['time'] = pd.to_datetime(df[time_col])
            df = df.sort_values('time').reset_index(drop=True)
            return df, f'🟡 CSV ({path})'
        except Exception:
            continue

    return None, '❌ No data source available'

# ============================================================================
# Load Models - from Scripts/models/
# ============================================================================

@st.cache_resource
def load_models():
    # Check these folders in order (handles case differences on Windows/Linux)
    candidate_dirs = [
        'Scripts/models',
        'scripts/models',
        'models',
        'models/advanced',
    ]

    model_dir = None
    for d in candidate_dirs:
        if os.path.isdir(d):
            model_dir = d
            break

    if model_dir is None:
        return None, None, f"❌ models folder not found. Checked: {candidate_dirs}"

    models  = {}
    scaler  = None

    # --- Find best model per horizon from results JSON ---
    results = None
    for fname in ['ml_only_results.json', 'ml_tuned_results.json',
                  'grid_search_results.json', 'results.json']:
        rpath = os.path.join(model_dir, fname)
        if os.path.exists(rpath):
            with open(rpath) as f:
                results = json.load(f)
            break

    for horizon in ['24h', '48h', '72h']:
        loaded = False

        # Try best model from JSON first
        if results and horizon in results:
            try:
                best_name = max(
                    results[horizon].items(),
                    key=lambda x: x[1].get('test_R2', x[1].get('R2', -999))
                )[0]
                fname = f"{best_name.lower().replace(' ', '_')}_{horizon}.pkl"
                fpath = os.path.join(model_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'rb') as f:
                        models[horizon] = pickle.load(f)
                    loaded = True
            except Exception:
                pass

        # Fallback: try each model name explicitly
        if not loaded:
            for name in ['xgboost', 'lightgbm', 'gradient_boosting',
                         'random_forest', 'ridge', 'lasso']:
                fpath = os.path.join(model_dir, f"{name}_{horizon}.pkl")
                if os.path.exists(fpath):
                    with open(fpath, 'rb') as f:
                        models[horizon] = pickle.load(f)
                    loaded = True
                    break

    # --- Load scaler ---
    for sname in ['scaler_ml.pkl', 'scaler.pkl', 'scaler_final.pkl']:
        spath = os.path.join(model_dir, sname)
        if os.path.exists(spath):
            with open(spath, 'rb') as f:
                scaler = pickle.load(f)
            break

    if not models:
        # List what's actually in the folder to help debug
        files = os.listdir(model_dir)
        return None, None, f"❌ No .pkl models found in {model_dir}. Files: {files[:10]}"

    if scaler is None:
        return None, None, f"❌ scaler_ml.pkl not found in {model_dir}"

    horizons_loaded = list(models.keys())
    return models, scaler, f"✅ {model_dir} | Horizons: {horizons_loaded}"

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.markdown("# 🌫️ Karachi AQI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔮 Predictions", "ℹ️ About"],
    label_visibility="collapsed"
)

# ============================================================================
# Load everything
# ============================================================================

data, data_source   = load_data()
models, scaler, model_status = load_models()

# Status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Status")
st.sidebar.markdown(f"**Data:** {data_source}")
st.sidebar.markdown(f"**Models:** {model_status}")

# Stop if critical data missing
if data is None:
    st.error("❌ No data loaded. Check MongoDB connection or CSV path.")
    st.stop()

if models is None:
    st.error(f"❌ {model_status}")
    st.info("Put your .pkl files in Scripts/models/ and scaler_ml.pkl must be there too.")
    st.stop()

# ============================================================================
# PAGE 1: Dashboard
# ============================================================================

if page == "📊 Dashboard":
    st.markdown('<p class="main-header">📊 Karachi AQI Dashboard</p>', unsafe_allow_html=True)

    current_aqi = float(data['aqi'].iloc[-1])
    category, color = get_aqi_category(current_aqi)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current AQI", f"{current_aqi:.0f}")
    with col2:
        st.markdown(f'<div class="metric-card">Status<br><span style="color:{color};font-weight:700">{category}</span></div>',
                    unsafe_allow_html=True)
    with col3:
        if 'pm2_5' in data.columns:
            st.metric("PM2.5", f"{float(data['pm2_5'].iloc[-1]):.1f} µg/m³")
    with col4:
        if 'temp' in data.columns:
            st.metric("Temperature", f"{float(data['temp'].iloc[-1]):.1f}°C")

    st.markdown("---")

    days_back   = st.slider("Show last N days", 7, 30, 14)
    cutoff      = data['time'].max() - timedelta(days=days_back)
    recent      = data[data['time'] >= cutoff].copy()

    # AQI Trend
    st.markdown('<p class="sub-header">AQI Trend</p>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['time'], y=recent['aqi'],
                             mode='lines', fill='tozeroy',
                             line=dict(color='#1f77b4', width=2)))
    fig.add_hrect(y0=0,   y1=50,  fillcolor="green",  opacity=0.08, line_width=0)
    fig.add_hrect(y0=50,  y1=100, fillcolor="yellow",  opacity=0.08, line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="orange",  opacity=0.08, line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="red",     opacity=0.08, line_width=0)
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="AQI",
                      hovermode='x unified', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Pollutants
    st.markdown('<p class="sub-header">Pollutant Levels</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_pm = go.Figure()
        if 'pm2_5' in recent.columns:
            fig_pm.add_trace(go.Scatter(x=recent['time'], y=recent['pm2_5'],
                                        mode='lines', name='PM2.5', line=dict(color='#ff7f0e')))
        if 'pm10' in recent.columns:
            fig_pm.add_trace(go.Scatter(x=recent['time'], y=recent['pm10'],
                                        mode='lines', name='PM10', line=dict(color='#2ca02c')))
        fig_pm.update_layout(height=300, xaxis_title="Date", yaxis_title="µg/m³",
                             hovermode='x unified')
        st.plotly_chart(fig_pm, use_container_width=True)

    with col2:
        fig_gas = go.Figure()
        if 'ozone' in recent.columns:
            fig_gas.add_trace(go.Scatter(x=recent['time'], y=recent['ozone'],
                                         mode='lines', name='Ozone', line=dict(color='#d62728')))
        if 'nitrogen_dioxide' in recent.columns:
            fig_gas.add_trace(go.Scatter(x=recent['time'], y=recent['nitrogen_dioxide'],
                                         mode='lines', name='NO₂', line=dict(color='#9467bd')))
        fig_gas.update_layout(height=300, xaxis_title="Date", yaxis_title="ppb",
                              hovermode='x unified')
        st.plotly_chart(fig_gas, use_container_width=True)

    # Weather
    st.markdown('<p class="sub-header">Weather Conditions</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'temp' in recent.columns:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=recent['time'], y=recent['temp'],
                                       mode='lines', fill='tozeroy', line=dict(color='#ff6b6b')))
            fig_t.update_layout(height=250, yaxis_title="Temp (°C)", showlegend=False)
            st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        if 'rhum' in recent.columns:
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=recent['time'], y=recent['rhum'],
                                       mode='lines', fill='tozeroy', line=dict(color='#4ecdc4')))
            fig_h.update_layout(height=250, yaxis_title="Humidity (%)", showlegend=False)
            st.plotly_chart(fig_h, use_container_width=True)

    with col3:
        if 'wspd' in recent.columns:
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(x=recent['time'], y=recent['wspd'],
                                       mode='lines', fill='tozeroy', line=dict(color='#95e1d3')))
            fig_w.update_layout(height=250, yaxis_title="Wind (km/h)", showlegend=False)
            st.plotly_chart(fig_w, use_container_width=True)

    # Statistics
    st.markdown('<p class="sub-header">Statistics (Last 7 Days)</p>', unsafe_allow_html=True)
    last7 = data[data['time'] >= data['time'].max() - timedelta(days=7)]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Avg AQI",        f"{last7['aqi'].mean():.1f}")
    with col2: st.metric("Max AQI",        f"{last7['aqi'].max():.0f}")
    with col3: st.metric("Min AQI",        f"{last7['aqi'].min():.0f}")
    with col4: st.metric("Good Hours",     f"{(last7['aqi'] <= 50).sum()}")
    with col5: st.metric("Unhealthy Hours",f"{(last7['aqi'] > 150).sum()}")

# ============================================================================
# PAGE 2: Predictions
# ============================================================================

elif page == "🔮 Predictions":
    st.markdown('<p class="main-header">🔮 AQI Predictions</p>', unsafe_allow_html=True)
    st.info("📌 Predictions made using your trained ML models from Scripts/models/")

    current_aqi = float(data['aqi'].iloc[-1])

    # Prepare features
    exclude = ['time', 'timestamp', 'aqi_24h', 'aqi_48h', 'aqi_72h',
               'dominant_pollutant', 'aqi_category', 'aqi_color', 'time_of_day']
    feat_cols = [c for c in data.columns if c not in exclude]
    latest = data[feat_cols].select_dtypes(include=[np.number]).iloc[-1:].fillna(0)

    # Align feature count with scaler
    n_expected = scaler.n_features_in_
    vals = latest.values
    if vals.shape[1] < n_expected:
        padded = np.zeros((1, n_expected))
        padded[0, :vals.shape[1]] = vals[0]
        vals = padded
    elif vals.shape[1] > n_expected:
        vals = vals[:, :n_expected]

    try:
        features_scaled = scaler.transform(vals)
    except Exception as e:
        st.error(f"❌ Scaling failed: {e}")
        st.stop()

    # Make predictions
    predictions = {}
    for horizon in ['24h', '48h', '72h']:
        hours = int(horizon.replace('h', ''))
        if horizon in models:
            try:
                pred_aqi = float(models[horizon].predict(features_scaled)[0])
            except Exception:
                pred_aqi = current_aqi
        else:
            pred_aqi = current_aqi

        pred_aqi = max(0, pred_aqi)  # AQI can't be negative
        category, color = get_aqi_category(pred_aqi)
        predictions[horizon] = {
            'aqi': pred_aqi,
            'time': datetime.now() + timedelta(hours=hours),
            'category': category,
            'color': color
        }

    # Display prediction cards
    st.markdown('<p class="sub-header">Forecast</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    for col, horizon in zip([col1, col2, col3], ['24h', '48h', '72h']):
        with col:
            pred = predictions[horizon]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1.5rem; border-radius: 1rem; color: white; margin-bottom: 1rem;">
                <h3 style="margin:0">{horizon.upper()} Ahead</h3>
                <p style="margin:0.4rem 0;opacity:0.9">{pred['time'].strftime('%b %d, %I:%M %p')}</p>
                <h1 style="margin:0.4rem 0;font-size:3rem">{pred['aqi']:.0f}</h1>
                <p style="margin:0;font-size:1.1rem">{pred['category']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Forecast chart
    st.markdown('<p class="sub-header">Historical + Forecast</p>', unsafe_allow_html=True)
    hist7 = data[data['time'] >= data['time'].max() - timedelta(days=7)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist7['time'], y=hist7['aqi'],
                             mode='lines', name='Historical',
                             line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(
        x=[hist7['time'].iloc[-1]] + [predictions[h]['time'] for h in ['24h', '48h', '72h']],
        y=[float(hist7['aqi'].iloc[-1])] + [predictions[h]['aqi'] for h in ['24h', '48h', '72h']],
        mode='lines+markers', name='Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    fig.update_layout(height=400, xaxis_title="Date & Time", yaxis_title="AQI",
                      hovermode='x unified',
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(fig, use_container_width=True)

    # AQI Distribution
    st.markdown('<p class="sub-header">AQI Category Distribution (Last 30 Days)</p>', unsafe_allow_html=True)
    last30 = data[data['time'] >= data['time'].max() - timedelta(days=30)]
    cats = [get_aqi_category(v)[0] for v in last30['aqi']]
    cat_counts = pd.Series(cats).value_counts()

    fig_pie = go.Figure(data=[go.Pie(
        labels=cat_counts.index, values=cat_counts.values, hole=0.4,
        marker=dict(colors=['#00e400','#ffff00','#ff7e00','#ff0000','#8f3f97','#7e0023']),
        textinfo='label+percent', textfont_size=13
    )])
    fig_pie.update_layout(height=380, showlegend=True,
                          legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    st.plotly_chart(fig_pie, use_container_width=True)

    # Health recommendations
    st.markdown('<p class="sub-header">Health Recommendations</p>', unsafe_allow_html=True)
    max_aqi = max(predictions[h]['aqi'] for h in ['24h', '48h', '72h'])
    if max_aqi <= 50:
        st.success("✅ Air quality is good. Enjoy outdoor activities!")
    elif max_aqi <= 100:
        st.info("ℹ️ Moderate. Sensitive individuals should limit prolonged outdoor exertion.")
    elif max_aqi <= 150:
        st.warning("⚠️ Unhealthy for sensitive groups. Limit outdoor exposure.")
    elif max_aqi <= 200:
        st.warning("⚠️ Unhealthy. Reduce prolonged outdoor exertion.")
    elif max_aqi <= 300:
        st.error("🚨 Very Unhealthy. Avoid outdoor activities.")
    else:
        st.error("☢️ Hazardous. Stay indoors. Use N95 masks if necessary.")

# ============================================================================
# PAGE 3: About
# ============================================================================

elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Purpose
    Real-time Air Quality Index (AQI) predictions for Karachi, Pakistan.

    ### 🤖 ML Models
    - XGBoost, LightGBM, Gradient Boosting, Random Forest, Ridge, Lasso
    - Trained with GridSearchCV hyperparameter tuning
    - Models stored in: `Scripts/models/`

    ### 📊 Data
    - **Primary**: MongoDB Atlas (`aqi_feature_store`)
    - **Fallback**: `data/cleaned_aqi_data_v2.csv`
    - **Update**: Daily via GitHub Actions

    ### 🔮 Predictions
    | Horizon | Description |
    |---------|-------------|
    | 24h | Next 24 hours |
    | 48h | Next 48 hours |
    | 72h | Next 72 hours |

    ### 🚀 Deployment
    - **Dashboard**: Streamlit
    - **CI/CD**: GitHub Actions (daily retraining)
    - **Hosting**: HuggingFace Spaces
    """)

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#666'>"
    f"Karachi AQI Dashboard | Data: {data_source} | "
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    f"</div>",
    unsafe_allow_html=True
)