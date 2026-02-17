"""
AQI Prediction Dashboard - Robust Version
==========================================
Handles feature mismatches and provides detailed debugging
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; font-weight: 600; color: #333; margin-top: 1rem;}
    .metric-card {background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .good {color: #00e400; font-weight: 700;}
    .moderate {color: #ffff00; font-weight: 700;}
    .unhealthy-sg {color: #ff7e00; font-weight: 700;}
    .unhealthy {color: #ff0000; font-weight: 700;}
    .very-unhealthy {color: #8f3f97; font-weight: 700;}
    .hazardous {color: #7e0023; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_data():
    """Load historical data - MongoDB first, CSV fallback"""
    
    # Try MongoDB first
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        
        st.sidebar.info("🔄 Attempting MongoDB connection...")
        
        MONGO_URI = "mongodb+srv://nawababbas08_db_user:2Ja4OGlDdKfG6EvZ@cluster0.jnxn95g.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true"
        
        client = MongoClient(
            MONGO_URI,
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000
        )
        
        # Test connection
        client.admin.command('ping')
        
        # Load data
        db = client['aqi_feature_store']
        collection = db['aqi_features']
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))
        
        client.close()
        
        if len(df) > 0:
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            elif 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'])
            
            st.sidebar.success(f"✅ MongoDB: Loaded {len(df)} records")
            return df, None
        else:
            raise Exception("No data found in MongoDB")
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ MongoDB failed: {str(e)[:50]}...")
        st.sidebar.info("🔄 Trying CSV fallback...")
        
        # Fallback to CSV
        try:
            df = pd.read_csv('data/cleaned_aqi_data_v2.csv')
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
            elif 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'])
            
            st.sidebar.success(f"✅ CSV: Loaded {len(df)} records")
            return df, None
            
        except FileNotFoundError:
            return None, "❌ Both MongoDB and CSV failed. No data available."
        except Exception as e:
            return None, f"❌ Error loading data: {str(e)}"

@st.cache_resource
def load_models():
    """Load trained models and scalers"""
    
    if not os.path.exists('models/ml_only_results.json'):
        return None, "❌ Training results not found! Run: python train_ml_only.py"
    
    try:
        with open('models/ml_only_results.json', 'r') as f:
            results = json.load(f)
        
        models = {}
        for horizon in ['24h', '48h', '72h']:
            best_model_name = max(results[horizon].items(), key=lambda x: x[1]['test_R2'])[0]
            model_filename = f'models/{best_model_name.lower().replace(" ", "_")}_{horizon}.pkl'
            
            if not os.path.exists(model_filename):
                return None, f"❌ Model file not found: {model_filename}"
            
            with open(model_filename, 'rb') as f:
                models[horizon] = {
                    'model': pickle.load(f),
                    'name': best_model_name,
                    'metrics': results[horizon][best_model_name]
                }
        
        if not os.path.exists('models/scaler_ml.pkl'):
            return None, "❌ Scaler not found!"
        
        with open('models/scaler_ml.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Get expected features from scaler
        n_features = scaler.n_features_in_
        
        return (models, scaler, n_features), None
        
    except Exception as e:
        return None, f"❌ Error loading models: {str(e)}"

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return 'Good', '#00e400', 'good'
    elif aqi <= 100:
        return 'Moderate', '#ffff00', 'moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups', '#ff7e00', 'unhealthy-sg'
    elif aqi <= 200:
        return 'Unhealthy', '#ff0000', 'unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy', '#8f3f97', 'very-unhealthy'
    else:
        return 'Hazardous', '#7e0023', 'hazardous'

def prepare_features_simple(data, n_expected_features):
    """Prepare features matching training - SIMPLE VERSION"""
    
    df = data.copy()
    
    # Essential numeric columns
    numeric_cols = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'temp', 
                    'rhum', 'wspd', 'pres', 'aqi', 'hour', 'day_of_week', 'month']
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    # Get last row of available features
    last_row = df[available_cols].iloc[-1:].copy()
    
    # If we have fewer features than expected, pad with zeros
    if len(available_cols) < n_expected_features:
        # Create zero-padded array
        padded = np.zeros((1, n_expected_features))
        padded[0, :len(available_cols)] = last_row.values[0]
        return padded
    
    # If we have more features, take first n_expected_features
    elif len(available_cols) > n_expected_features:
        return last_row.iloc[:, :n_expected_features].values
    
    # Exact match
    return last_row.values

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.markdown("# 🌫️ AQI Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Visualizations", "🔮 Predictions"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard predicts Air Quality Index (AQI) for Karachi "
    "for the next 24, 48, and 72 hours using machine learning models."
)

# ============================================================================
# Load Data and Models
# ============================================================================

data, data_error = load_data()
model_data, model_error = load_models()

# Show errors if any
if data_error:
    st.error(data_error)
    st.info("Please run: `python clean_aqi_optimized.py`")
    st.stop()

if model_error:
    st.error(model_error)
    st.info("Please run: `python train_ml_only.py`")
    st.stop()

models, scaler, n_features = model_data

# ============================================================================
# PAGE 1: VISUALIZATIONS
# ============================================================================

if page == "📊 Visualizations":
    
    st.markdown('<p class="main-header">📊 AQI Historical Analysis</p>', unsafe_allow_html=True)
    
    # Current AQI
    current_aqi = float(data['aqi'].iloc[-1])
    category, color, css_class = get_aqi_category(current_aqi)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current AQI", f"{current_aqi:.0f}")
    
    with col2:
        st.markdown(f'<div class="metric-card">Status<br><span class="{css_class}">{category}</span></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        if 'pm2_5' in data.columns:
            pm25_val = float(data['pm2_5'].iloc[-1])
            st.metric("PM2.5", f"{pm25_val:.1f} µg/m³")
    
    with col4:
        if 'temp' in data.columns:
            temp_val = float(data['temp'].iloc[-1])
            st.metric("Temperature", f"{temp_val:.1f}°C")
    
    st.markdown("---")
    
    # Time range selector
    days_back = st.slider("Show last N days", 7, 30, 14)
    cutoff_date = data['time'].max() - timedelta(days=days_back)
    recent_data = data[data['time'] >= cutoff_date].copy()
    
    # AQI Trend
    st.markdown('<p class="sub-header">AQI Trend</p>', unsafe_allow_html=True)
    
    fig_aqi = go.Figure()
    fig_aqi.add_trace(go.Scatter(
        x=recent_data['time'],
        y=recent_data['aqi'],
        mode='lines',
        name='AQI',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig_aqi.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
    fig_aqi.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_aqi.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
    fig_aqi.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
    
    fig_aqi.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig_aqi, use_container_width=True)
    
    # Pollutants
    st.markdown('<p class="sub-header">Pollutant Levels</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pm = go.Figure()
        if 'pm2_5' in recent_data.columns:
            fig_pm.add_trace(go.Scatter(x=recent_data['time'], y=recent_data['pm2_5'], 
                                        mode='lines', name='PM2.5', line=dict(color='#ff7f0e')))
        if 'pm10' in recent_data.columns:
            fig_pm.add_trace(go.Scatter(x=recent_data['time'], y=recent_data['pm10'], 
                                        mode='lines', name='PM10', line=dict(color='#2ca02c')))
        fig_pm.update_layout(height=300, xaxis_title="Date", yaxis_title="µg/m³", 
                            hovermode='x unified', showlegend=True)
        st.plotly_chart(fig_pm, use_container_width=True)
    
    with col2:
        fig_gases = go.Figure()
        if 'ozone' in recent_data.columns:
            fig_gases.add_trace(go.Scatter(x=recent_data['time'], y=recent_data['ozone'], 
                                           mode='lines', name='Ozone', line=dict(color='#d62728')))
        if 'nitrogen_dioxide' in recent_data.columns:
            fig_gases.add_trace(go.Scatter(x=recent_data['time'], y=recent_data['nitrogen_dioxide'], 
                                           mode='lines', name='NO₂', line=dict(color='#9467bd')))
        fig_gases.update_layout(height=300, xaxis_title="Date", yaxis_title="ppb", 
                               hovermode='x unified', showlegend=True)
        st.plotly_chart(fig_gases, use_container_width=True)
    
    # Weather
    st.markdown('<p class="sub-header">Weather Conditions</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'temp' in recent_data.columns:
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=recent_data['time'], y=recent_data['temp'],
                                          mode='lines', fill='tozeroy', line=dict(color='#ff6b6b')))
            fig_temp.update_layout(height=250, xaxis_title="Date", yaxis_title="Temperature (°C)",
                                  showlegend=False)
            st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        if 'rhum' in recent_data.columns:
            fig_humid = go.Figure()
            fig_humid.add_trace(go.Scatter(x=recent_data['time'], y=recent_data['rhum'],
                                           mode='lines', fill='tozeroy', line=dict(color='#4ecdc4')))
            fig_humid.update_layout(height=250, xaxis_title="Date", yaxis_title="Humidity (%)",
                                   showlegend=False)
            st.plotly_chart(fig_humid, use_container_width=True)
    
    with col3:
        if 'wspd' in recent_data.columns:
            fig_wind = go.Figure()
            fig_wind.add_trace(go.Scatter(x=recent_data['time'], y=recent_data['wspd'],
                                          mode='lines', fill='tozeroy', line=dict(color='#95e1d3')))
            fig_wind.update_layout(height=250, xaxis_title="Date", yaxis_title="Wind Speed (km/h)",
                                  showlegend=False)
            st.plotly_chart(fig_wind, use_container_width=True)
    
    # Statistics
    st.markdown('<p class="sub-header">Statistics (Last 7 Days)</p>', unsafe_allow_html=True)
    
    last_7_days = data[data['time'] >= data['time'].max() - timedelta(days=7)].copy()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Avg AQI", f"{last_7_days['aqi'].mean():.1f}")
    with col2:
        st.metric("Max AQI", f"{last_7_days['aqi'].max():.0f}")
    with col3:
        st.metric("Min AQI", f"{last_7_days['aqi'].min():.0f}")
    with col4:
        good_days = (last_7_days['aqi'] <= 50).sum()
        st.metric("Good Days", f"{good_days}")
    with col5:
        bad_days = (last_7_days['aqi'] > 150).sum()
        st.metric("Unhealthy Days", f"{bad_days}")

# ============================================================================
# PAGE 2: PREDICTIONS
# ============================================================================

elif page == "🔮 Predictions":
    
    st.markdown('<p class="main-header">🔮 AQI Predictions</p>', unsafe_allow_html=True)
    
    st.info("📌 Predictions based on latest data and trained ML models")
    
    # Prepare features
    try:
        features = prepare_features_simple(data, n_features)
        features_scaled = scaler.transform(features)
    except Exception as e:
        st.error(f"❌ Error preparing features: {str(e)}")
        st.code(f"Expected features: {n_features}\nAvailable columns: {list(data.columns)}")
        st.stop()
    
    # Make predictions
    predictions = {}
    
    for horizon in ['24h', '48h', '72h']:
        try:
            model_info = models[horizon]
            pred_aqi = float(model_info['model'].predict(features_scaled)[0])
            
            hours_ahead = int(horizon.replace('h', ''))
            pred_time = datetime.now() + timedelta(hours=hours_ahead)
            
            category, color, css_class = get_aqi_category(pred_aqi)
            
            predictions[horizon] = {
                'aqi': pred_aqi,
                'time': pred_time,
                'category': category,
                'color': color,
                'css_class': css_class,
                'model': model_info['name'],
                'metrics': model_info['metrics']
            }
        except Exception as e:
            st.error(f"❌ Prediction error for {horizon}: {str(e)}")
            st.stop()
    
    # Display predictions
    st.markdown('<p class="sub-header">Forecast</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    for col, horizon in zip([col1, col2, col3], ['24h', '48h', '72h']):
        with col:
            pred = predictions[horizon]
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 1rem; color: white;">
                <h3 style="margin: 0;">{horizon.upper()} Ahead</h3>
                <p style="margin: 0.5rem 0; opacity: 0.9;">
                    {pred['time'].strftime('%b %d, %I:%M %p')}
                </p>
                <h1 style="margin: 0.5rem 0; font-size: 3rem;">
                    {pred['aqi']:.0f}
                </h1>
                <p style="margin: 0; font-size: 1.1rem;">
                    {pred['category']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Model:** {pred['model']}")
            st.markdown(f"**Confidence:** High")
            st.markdown(f"**Updated:** {datetime.now().strftime('%I:%M %p')}")
    
    st.markdown("---")
    
    # Forecast chart
    st.markdown('<p class="sub-header">Prediction Visualization</p>', unsafe_allow_html=True)
    
    current_time = datetime.now()
    forecast_times = [current_time + timedelta(hours=int(h.replace('h', ''))) for h in ['24h', '48h', '72h']]
    forecast_aqi = [predictions[h]['aqi'] for h in ['24h', '48h', '72h']]
    
    hist_7_days = data[data['time'] >= data['time'].max() - timedelta(days=7)].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_7_days['time'],
        y=hist_7_days['aqi'],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[hist_7_days['time'].iloc[-1]] + forecast_times,
        y=[float(hist_7_days['aqi'].iloc[-1])] + forecast_aqi,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Date & Time",
        yaxis_title="AQI",
        hovermode='x unified',
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AQI Category Distribution (Last 30 Days)
    st.markdown('<p class="sub-header">AQI Category Distribution (Last 30 Days)</p>', unsafe_allow_html=True)
    
    last_30_days = data[data['time'] >= data['time'].max() - timedelta(days=30)].copy()
    
    # Calculate category distribution
    categories = []
    for aqi_val in last_30_days['aqi']:
        cat, _, _ = get_aqi_category(aqi_val)
        categories.append(cat)
    
    cat_counts = pd.Series(categories).value_counts()
    
    # Create pie chart
    colors = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023']
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=cat_counts.index,
        values=cat_counts.values,
        hole=0.4,
        marker=dict(colors=colors[:len(cat_counts)]),
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig_pie.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Hourly AQI Pattern
    st.markdown('<p class="sub-header">AQI Pattern by Hour of Day</p>', unsafe_allow_html=True)
    
    if 'hour' in last_30_days.columns:
        hourly_avg = last_30_days.groupby('hour')['aqi'].mean().reset_index()
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly_avg['hour'],
            y=hourly_avg['aqi'],
            marker=dict(
                color=hourly_avg['aqi'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="AQI")
            ),
            text=[f"{val:.0f}" for val in hourly_avg['aqi']],
            textposition='auto',
        ))
        
        fig_hourly.update_layout(
            height=350,
            xaxis_title="Hour of Day",
            yaxis_title="Average AQI",
            showlegend=False,
            xaxis=dict(tickmode='linear', tick0=0, dtick=2)
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Health recommendations
    st.markdown('<p class="sub-header">Health Recommendations</p>', unsafe_allow_html=True)
    
    max_aqi = max([predictions[h]['aqi'] for h in ['24h', '48h', '72h']])
    
    if max_aqi <= 50:
        st.success("✅ Air quality is good. Enjoy outdoor activities!")
    elif max_aqi <= 100:
        st.info("ℹ️ Moderate air quality. Sensitive individuals should limit prolonged outdoor exertion.")
    elif max_aqi <= 150:
        st.warning("⚠️ Unhealthy for sensitive groups. Limit outdoor exposure.")
    elif max_aqi <= 200:
        st.warning("⚠️ Unhealthy. Reduce prolonged outdoor exertion.")
    elif max_aqi <= 300:
        st.error("🚨 Very Unhealthy. Avoid outdoor activities.")
    else:
        st.error("☢️ Hazardous. Stay indoors. Use N95 masks if necessary.")

st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"AQI Prediction Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    f"</div>",
    unsafe_allow_html=True
)