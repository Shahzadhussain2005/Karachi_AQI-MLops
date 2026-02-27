# 🌫️ Karachi Air Quality Index (AQI) Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time Air Quality Index prediction system for Karachi, Pakistan using Machine Learning with automated daily retraining via CI/CD pipeline.

![Dashboard Preview](assets/dashboard.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project provides **multi-horizon air quality forecasts** (24h, 48h, 72h) for Karachi using state-of-the-art machine learning models. The system:

- Fetches real-time AQI data from multiple sources
- Engineers 50+ features including lag, rolling statistics, and cyclical time features
- Trains optimized models daily via automated CI/CD pipeline
- Delivers predictions through an interactive Streamlit dashboard

**Live Demo**: [Coming Soon]

---

## ✨ Features

### 📊 Dashboard
- Real-time AQI monitoring
- Historical trend visualization (7-30 days)
- Pollutant levels tracking (PM2.5, PM10, O₃, NO₂)
- Weather conditions display (Temperature, Humidity, Wind)
- AQI category distribution
- 7-day statistics summary

### 🔮 Predictions
- Multi-horizon forecasts: 24h, 48h, 72h
- Confidence-based health recommendations
- Interactive historical + forecast charts
- Model performance metrics

### 🤖 ML Models
- **24h predictions**: XGBoost (hyperparameter-tuned)
- **48h predictions**: LightGBM (hyperparameter-tuned)
- **72h predictions**: Ridge Regression (regularized)
- **Feature engineering**: 50+ features with lag, rolling, and cyclical transformations
- **Validation**: Time-series cross-validation
- **Performance**: R² scores of 0.65-0.80 (24h), 0.55-0.70 (48h), 0.45-0.60 (72h)

### ⚙️ Automation
- Daily data fetching from APIs
- Automated model retraining via GitHub Actions
- MongoDB Atlas for persistent data storage
- Model versioning and tracking

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────┤
│  Open-Meteo API  │  AQICN API  │  Meteostat  │  User Uploads   │
└────────┬──────────────────┬──────────────┬─────────────────────┘
         │                  │              │
         v                  v              v
┌─────────────────────────────────────────────────────────────────┐
│                   DATA COLLECTION & CLEANING                     │
│  • Scripts/Fetch_latest_data.ipynb                              │
│  • Scripts/clean_data.ipynb                                     │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                      MONGODB ATLAS                               │
│  Database: aqi_feature_store                                    │
│  Collection: aqi_features                                       │
│  • 4000+ historical records                                     │
│  • Real-time updates                                            │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
│  • Lag features (1h, 3h, 6h, 12h, 24h, 48h)                    │
│  • Rolling statistics (mean, std, min, max)                     │
│  • Difference features (trend detection)                        │
│  • Cyclical encoding (hour, day, month)                        │
│  • Total: 50-60 engineered features                            │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                              │
│  • XGBoost (24h): RandomizedSearchCV                           │
│  • LightGBM (48h): RandomizedSearchCV                          │
│  • Ridge (72h): GridSearchCV                                    │
│  • Validation: TimeSeriesSplit CV (k=2)                        │
│  • Scaling: RobustScaler                                       │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                   GITHUB ACTIONS CI/CD                           │
│  • Daily automated retraining (00:00 UTC)                      │
│  • Manual trigger available                                     │
│  • Model versioning and artifact storage                       │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                            │
│  • Real-time predictions                                        │
│  • Interactive visualizations                                   │
│  • Health recommendations                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend & ML
- **Python 3.11+**
- **scikit-learn** - Model training, preprocessing, validation
- **XGBoost** - Gradient boosting for 24h predictions
- **LightGBM** - Fast gradient boosting for 48h predictions
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Frontend & Visualization
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts and graphs

### Data & Storage
- **MongoDB Atlas** - Cloud database for feature store
- **pymongo** - MongoDB driver for Python

### DevOps & CI/CD
- **GitHub Actions** - Automated training pipeline
- **Docker** - Containerization (optional)

### APIs
- **Open-Meteo** - Air quality data (PM2.5, PM10, O₃, NO₂, SO₂, CO)
- **AQICN** - Current AQI readings
- **Meteostat** - Historical weather data

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- MongoDB Atlas account (free tier works)
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Shahzadhussain2005/Karachi_AQI-MLOps.git
cd Karachi_AQI-MLOps
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

5. **Run data collection (optional)**
```bash
cd Scripts
jupyter nbconvert --to python --execute Fetch_latest_data.ipynb
jupyter nbconvert --to python --execute clean_data.ipynb
```

6. **Train models**
```bash
jupyter nbconvert --to python --execute train_models.ipynb
```

7. **Launch dashboard**
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 🚀 Usage

### Running the Dashboard Locally

```bash
streamlit run app.py
```

### Manual Data Collection

```bash
cd Scripts
jupyter nbconvert --to python --execute Fetch_latest_data.ipynb
```

### Manual Model Training

```bash
cd Scripts
jupyter nbconvert --to python --execute train_models.ipynb
```

### Viewing Logs

Training logs and model performance metrics are saved in:
- `models/results.json` - Performance scores
- `models/feature_names.json` - List of features used

---

## 📁 Project Structure

```
Karachi_AQI-MLOps/
├── .github/
│   └── workflows/
│       └── daily_retrain.yml          # CI/CD pipeline
├── Scripts/
│   ├── Fetch_latest_data.ipynb        # Data collection
│   ├── clean_data.ipynb               # Data preprocessing
│   ├── train_models.ipynb             # Model training
│   ├── mongodb_connect.ipynb          # MongoDB upload
│   └── models/                        # Trained models
│       ├── xgboost_24h.pkl
│       ├── lightgbm_48h.pkl
│       ├── ridge_72h.pkl
│       ├── scaler_ml.pkl
│       ├── feature_names.json
│       └── results.json
├── data/
│   └── cleaned_aqi_data_v2.csv        # Fallback CSV data
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables (create this)
├── .gitignore                         # Git ignore rules
├── Dockerfile                         # Docker configuration (optional)
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## 🤖 ML Pipeline

### 1. Data Collection

**Sources:**
- **Open-Meteo API**: PM2.5, PM10, NO₂, O₃, SO₂, CO
- **AQICN API**: Current AQI readings
- **Meteostat**: Temperature, humidity, wind speed, pressure

**Frequency:** Every hour (automated via GitHub Actions)

### 2. Feature Engineering

```python
# Lag features
aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h, aqi_lag_48h

# Rolling statistics
aqi_ma_3h, aqi_ma_6h, aqi_ma_12h, aqi_ma_24h
aqi_std_3h, aqi_std_6h, aqi_std_12h, aqi_std_24h
aqi_min_3h, aqi_min_6h, aqi_min_12h, aqi_min_24h
aqi_max_3h, aqi_max_6h, aqi_max_12h, aqi_max_24h

# Difference features
aqi_diff_1h, aqi_diff_3h, aqi_diff_24h

# Cyclical encoding
hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

# Total: 50-60 features
```

### 3. Model Training

**XGBoost (24h predictions)**
```python
XGBRegressor(
    n_estimators=50-200,     # Tuned
    max_depth=3-7,           # Tuned
    learning_rate=0.01-0.1,  # Tuned
    subsample=0.8-1.0,       # Tuned
    random_state=42
)
```

**LightGBM (48h predictions)**
```python
LGBMRegressor(
    n_estimators=50-200,     # Tuned
    max_depth=3-7,           # Tuned
    learning_rate=0.01-0.1,  # Tuned
    num_leaves=31-63,        # Tuned
    random_state=42
)
```

**Ridge (72h predictions)**
```python
Ridge(
    alpha=0.1-100.0,         # Tuned
    solver='auto'/'svd'/'saga'  # Tuned
)
```

### 4. Evaluation

**Metrics:**
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Accuracy ±20**: Predictions within 20 AQI units

**Validation:**
- Time-series cross-validation (TimeSeriesSplit, k=2)
- 80/20 train-test split (chronological order maintained)

### 5. Deployment

Models are saved as:
- `xgboost_24h.pkl`
- `lightgbm_48h.pkl`
- `ridge_72h.pkl`
- `scaler_ml.pkl`
- `feature_names.json`

---

## 📡 API Reference

### Open-Meteo API

**Endpoint:**
```
https://air-quality-api.open-meteo.com/v1/air-quality
```

**Parameters:**
```python
params = {
    'latitude': 24.8607,
    'longitude': 67.0011,
    'hourly': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone',
    'past_days': 180
}
```

### AQICN API

**Endpoint:**
```
https://api.waqi.info/feed/karachi/
```

**Parameters:**
```python
params = {'token': 'YOUR_TOKEN'}
```

### MongoDB Atlas

**Connection:**
```python
from pymongo import MongoClient
from pymongo.server_api import ServerApi

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi('1')
)
db = client['aqi_feature_store']
collection = db['aqi_features']
```

---

## 🌐 Deployment

### GitHub Actions CI/CD

The project includes automated daily retraining:

**Workflow:** `.github/workflows/daily_retrain.yml`

**Schedule:** Every day at 00:00 UTC (cron: `'0 0 * * *'`)

**Steps:**
1. Fetch latest data from APIs
2. Clean and engineer features
3. Upload to MongoDB Atlas
4. Train models with hyperparameter tuning
5. Save models to `models/` directory
6. Commit and push to GitHub

**Manual Trigger:**
```bash
# Go to GitHub Actions tab
# Click "Daily Model Retraining"
# Click "Run workflow"
```

### Required Secrets

Set these in **GitHub Settings → Secrets → Actions**:

```
MONGODB_URI    # MongoDB connection string
GH_PAT         # GitHub Personal Access Token (for push access)
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set `MONGODB_URI` in Streamlit secrets
5. Deploy!

### Docker Deployment (Optional)

```bash
# Build image
docker build -t karachi-aqi .

# Run container
docker run -p 8501:8501 \
  -e MONGODB_URI="your_mongodb_uri" \
  karachi-aqi
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/

# APIs (optional, for data collection)
AQICN_TOKEN=your_token_here
```

### Model Hyperparameters

Edit hyperparameter search spaces in `daily_retrain.yml`:

```python
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    # Add more parameters...
}
```

---

## 📊 Performance

### Model Scores (Latest)

| Model | Horizon | R² Score | RMSE | MAE | Acc ±20 |
|-------|---------|----------|------|-----|---------|
| XGBoost | 24h | 0.72 | 42.5 | 28.3 | 72% |
| LightGBM | 48h | 0.64 | 48.2 | 32.1 | 68% |
| Ridge | 72h | 0.53 | 54.8 | 36.4 | 61% |

*Scores updated: 2025-02-28*

### Feature Importance (Top 10)

1. `aqi_lag_24h` - AQI 24 hours ago
2. `aqi_ma_24h` - 24-hour moving average
3. `pm2_5` - Current PM2.5 level
4. `aqi_lag_12h` - AQI 12 hours ago
5. `aqi_std_24h` - 24-hour standard deviation
6. `day_of_year` - Seasonal patterns
7. `aqi_min_24h` - 24-hour minimum
8. `pm25_lag_24h` - PM2.5 24 hours ago
9. `temp` - Temperature
10. `aqi_diff_24h` - 24-hour AQI change

---

## 🧪 Testing

### Run Tests

```bash
# Test data loading
python -c "from app import load_data; print(load_data())"

# Test model loading
python -c "from app import load_models_and_features; print(load_models_and_features())"

# Test predictions
python DIAGNOSE.py
```

### Performance Benchmarks

```bash
# Measure prediction latency
python -c "
import time
from app import load_models_and_features
models, scaler, features, _ = load_models_and_features()
start = time.time()
# [prediction code]
print(f'Latency: {(time.time()-start)*1000:.2f}ms')
"
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Write tests for new features
- Update README for significant changes

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Shahzad Hussain**

📧 Email: shahzadhussain9680@gmail.com  
🔗 GitHub: [@Shahzadhussain2005](https://github.com/Shahzadhussain2005)  
💼 LinkedIn: [Add your LinkedIn]

---

## 🙏 Acknowledgments

- **Open-Meteo** for free air quality API
- **AQICN** for real-time AQI data
- **MongoDB Atlas** for cloud database
- **Streamlit** for rapid dashboard development
- **GitHub Actions** for free CI/CD

---

## 📈 Roadmap

- [ ] Add email/SMS alerts for high AQI
- [ ] Implement SHAP explainability
- [ ] Add multi-city support
- [ ] Mobile app development
- [ ] Real-time API endpoint
- [ ] Historical data export
- [ ] Comparison with other cities

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Shahzadhussain2005/Karachi_AQI-MLOps&type=Date)](https://star-history.com/#Shahzadhussain2005/Karachi_AQI-MLOps&Date)

---

<div align="center">

**Made with ❤️ for Karachi**

If you find this project useful, please consider giving it a ⭐!

</div># 🌫️ Karachi Air Quality Index (AQI) Prediction System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time Air Quality Index prediction system for Karachi, Pakistan using Machine Learning with automated daily retraining via CI/CD pipeline.

![Dashboard Preview](assets/dashboard.png)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project provides **multi-horizon air quality forecasts** (24h, 48h, 72h) for Karachi using state-of-the-art machine learning models. The system:

- Fetches real-time AQI data from multiple sources
- Engineers 50+ features including lag, rolling statistics, and cyclical time features
- Trains optimized models daily via automated CI/CD pipeline
- Delivers predictions through an interactive Streamlit dashboard

**Live Demo**: [Coming Soon]

---

## ✨ Features

### 📊 Dashboard
- Real-time AQI monitoring
- Historical trend visualization (7-30 days)
- Pollutant levels tracking (PM2.5, PM10, O₃, NO₂)
- Weather conditions display (Temperature, Humidity, Wind)
- AQI category distribution
- 7-day statistics summary

### 🔮 Predictions
- Multi-horizon forecasts: 24h, 48h, 72h
- Confidence-based health recommendations
- Interactive historical + forecast charts
- Model performance metrics

### 🤖 ML Models
- **24h predictions**: XGBoost (hyperparameter-tuned)
- **48h predictions**: LightGBM (hyperparameter-tuned)
- **72h predictions**: Ridge Regression (regularized)
- **Feature engineering**: 50+ features with lag, rolling, and cyclical transformations
- **Validation**: Time-series cross-validation
- **Performance**: R² scores of 0.65-0.80 (24h), 0.55-0.70 (48h), 0.45-0.60 (72h)

### ⚙️ Automation
- Daily data fetching from APIs
- Automated model retraining via GitHub Actions
- MongoDB Atlas for persistent data storage
- Model versioning and tracking

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
├─────────────────────────────────────────────────────────────────┤
│  Open-Meteo API  │  AQICN API  │  Meteostat  │  User Uploads   │
└────────┬──────────────────┬──────────────┬─────────────────────┘
         │                  │              │
         v                  v              v
┌─────────────────────────────────────────────────────────────────┐
│                   DATA COLLECTION & CLEANING                     │
│  • Scripts/Fetch_latest_data.ipynb                              │
│  • Scripts/clean_data.ipynb                                     │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                      MONGODB ATLAS                               │
│  Database: aqi_feature_store                                    │
│  Collection: aqi_features                                       │
│  • 4000+ historical records                                     │
│  • Real-time updates                                            │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                            │
│  • Lag features (1h, 3h, 6h, 12h, 24h, 48h)                    │
│  • Rolling statistics (mean, std, min, max)                     │
│  • Difference features (trend detection)                        │
│  • Cyclical encoding (hour, day, month)                        │
│  • Total: 50-60 engineered features                            │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                              │
│  • XGBoost (24h): RandomizedSearchCV                           │
│  • LightGBM (48h): RandomizedSearchCV                          │
│  • Ridge (72h): GridSearchCV                                    │
│  • Validation: TimeSeriesSplit CV (k=2)                        │
│  • Scaling: RobustScaler                                       │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                   GITHUB ACTIONS CI/CD                           │
│  • Daily automated retraining (00:00 UTC)                      │
│  • Manual trigger available                                     │
│  • Model versioning and artifact storage                       │
└────────┬────────────────────────────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD                            │
│  • Real-time predictions                                        │
│  • Interactive visualizations                                   │
│  • Health recommendations                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend & ML
- **Python 3.11+**
- **scikit-learn** - Model training, preprocessing, validation
- **XGBoost** - Gradient boosting for 24h predictions
- **LightGBM** - Fast gradient boosting for 48h predictions
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Frontend & Visualization
- **Streamlit** - Interactive web dashboard
- **Plotly** - Interactive charts and graphs

### Data & Storage
- **MongoDB Atlas** - Cloud database for feature store
- **pymongo** - MongoDB driver for Python

### DevOps & CI/CD
- **GitHub Actions** - Automated training pipeline
- **Docker** - Containerization (optional)

### APIs
- **Open-Meteo** - Air quality data (PM2.5, PM10, O₃, NO₂, SO₂, CO)
- **AQICN** - Current AQI readings
- **Meteostat** - Historical weather data

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- MongoDB Atlas account (free tier works)
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/Shahzadhussain2005/Karachi_AQI-MLOps.git
cd Karachi_AQI-MLOps
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the root directory:
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

5. **Run data collection (optional)**
```bash
cd Scripts
jupyter nbconvert --to python --execute Fetch_latest_data.ipynb
jupyter nbconvert --to python --execute clean_data.ipynb
```

6. **Train models**
```bash
jupyter nbconvert --to python --execute train_models.ipynb
```

7. **Launch dashboard**
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

---

## 🚀 Usage

### Running the Dashboard Locally

```bash
streamlit run app.py
```

### Manual Data Collection

```bash
cd Scripts
jupyter nbconvert --to python --execute Fetch_latest_data.ipynb
```

### Manual Model Training

```bash
cd Scripts
jupyter nbconvert --to python --execute train_models.ipynb
```

### Viewing Logs

Training logs and model performance metrics are saved in:
- `models/results.json` - Performance scores
- `models/feature_names.json` - List of features used

---

## 📁 Project Structure

```
Karachi_AQI-MLOps/
├── .github/
│   └── workflows/
│       └── daily_retrain.yml          # CI/CD pipeline
├── Scripts/
│   ├── Fetch_latest_data.ipynb        # Data collection
│   ├── clean_data.ipynb               # Data preprocessing
│   ├── train_models.ipynb             # Model training
│   ├── mongodb_connect.ipynb          # MongoDB upload
│   └── models/                        # Trained models
│       ├── xgboost_24h.pkl
│       ├── lightgbm_48h.pkl
│       ├── ridge_72h.pkl
│       ├── scaler_ml.pkl
│       ├── feature_names.json
│       └── results.json
├── data/
│   └── cleaned_aqi_data_v2.csv        # Fallback CSV data
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables (create this)
├── .gitignore                         # Git ignore rules
├── Dockerfile                         # Docker configuration (optional)
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## 🤖 ML Pipeline

### 1. Data Collection

**Sources:**
- **Open-Meteo API**: PM2.5, PM10, NO₂, O₃, SO₂, CO
- **AQICN API**: Current AQI readings
- **Meteostat**: Temperature, humidity, wind speed, pressure

**Frequency:** Every hour (automated via GitHub Actions)

### 2. Feature Engineering

```python
# Lag features
aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_12h, aqi_lag_24h, aqi_lag_48h

# Rolling statistics
aqi_ma_3h, aqi_ma_6h, aqi_ma_12h, aqi_ma_24h
aqi_std_3h, aqi_std_6h, aqi_std_12h, aqi_std_24h
aqi_min_3h, aqi_min_6h, aqi_min_12h, aqi_min_24h
aqi_max_3h, aqi_max_6h, aqi_max_12h, aqi_max_24h

# Difference features
aqi_diff_1h, aqi_diff_3h, aqi_diff_24h

# Cyclical encoding
hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

# Total: 50-60 features
```

### 3. Model Training

**XGBoost (24h predictions)**
```python
XGBRegressor(
    n_estimators=50-200,     # Tuned
    max_depth=3-7,           # Tuned
    learning_rate=0.01-0.1,  # Tuned
    subsample=0.8-1.0,       # Tuned
    random_state=42
)
```

**LightGBM (48h predictions)**
```python
LGBMRegressor(
    n_estimators=50-200,     # Tuned
    max_depth=3-7,           # Tuned
    learning_rate=0.01-0.1,  # Tuned
    num_leaves=31-63,        # Tuned
    random_state=42
)
```

**Ridge (72h predictions)**
```python
Ridge(
    alpha=0.1-100.0,         # Tuned
    solver='auto'/'svd'/'saga'  # Tuned
)
```

### 4. Evaluation

**Metrics:**
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Accuracy ±20**: Predictions within 20 AQI units

**Validation:**
- Time-series cross-validation (TimeSeriesSplit, k=2)
- 80/20 train-test split (chronological order maintained)

### 5. Deployment

Models are saved as:
- `xgboost_24h.pkl`
- `lightgbm_48h.pkl`
- `ridge_72h.pkl`
- `scaler_ml.pkl`
- `feature_names.json`

---

## 📡 API Reference

### Open-Meteo API

**Endpoint:**
```
https://air-quality-api.open-meteo.com/v1/air-quality
```

**Parameters:**
```python
params = {
    'latitude': 24.8607,
    'longitude': 67.0011,
    'hourly': 'pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone',
    'past_days': 180
}
```

### AQICN API

**Endpoint:**
```
https://api.waqi.info/feed/karachi/
```

**Parameters:**
```python
params = {'token': 'YOUR_TOKEN'}
```

### MongoDB Atlas

**Connection:**
```python
from pymongo import MongoClient
from pymongo.server_api import ServerApi

client = MongoClient(
    MONGODB_URI,
    server_api=ServerApi('1')
)
db = client['aqi_feature_store']
collection = db['aqi_features']
```

---

## 🌐 Deployment

### GitHub Actions CI/CD

The project includes automated daily retraining:

**Workflow:** `.github/workflows/daily_retrain.yml`

**Schedule:** Every day at 00:00 UTC (cron: `'0 0 * * *'`)

**Steps:**
1. Fetch latest data from APIs
2. Clean and engineer features
3. Upload to MongoDB Atlas
4. Train models with hyperparameter tuning
5. Save models to `models/` directory
6. Commit and push to GitHub

**Manual Trigger:**
```bash
# Go to GitHub Actions tab
# Click "Daily Model Retraining"
# Click "Run workflow"
```

### Required Secrets

Set these in **GitHub Settings → Secrets → Actions**:

```
MONGODB_URI    # MongoDB connection string
GH_PAT         # GitHub Personal Access Token (for push access)
```

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set `MONGODB_URI` in Streamlit secrets
5. Deploy!

### Docker Deployment (Optional)

```bash
# Build image
docker build -t karachi-aqi .

# Run container
docker run -p 8501:8501 \
  -e MONGODB_URI="your_mongodb_uri" \
  karachi-aqi
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# MongoDB
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/

# APIs (optional, for data collection)
AQICN_TOKEN=your_token_here
```

### Model Hyperparameters

Edit hyperparameter search spaces in `daily_retrain.yml`:

```python
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    # Add more parameters...
}
```

---

## 📊 Performance

### Model Scores (Latest)

| Model | Horizon | R² Score | RMSE | MAE | Acc ±20 |
|-------|---------|----------|------|-----|---------|
| XGBoost | 24h | 0.72 | 42.5 | 28.3 | 72% |
| LightGBM | 48h | 0.64 | 48.2 | 32.1 | 68% |
| Ridge | 72h | 0.53 | 54.8 | 36.4 | 61% |

*Scores updated: 2025-02-28*

### Feature Importance (Top 10)

1. `aqi_lag_24h` - AQI 24 hours ago
2. `aqi_ma_24h` - 24-hour moving average
3. `pm2_5` - Current PM2.5 level
4. `aqi_lag_12h` - AQI 12 hours ago
5. `aqi_std_24h` - 24-hour standard deviation
6. `day_of_year` - Seasonal patterns
7. `aqi_min_24h` - 24-hour minimum
8. `pm25_lag_24h` - PM2.5 24 hours ago
9. `temp` - Temperature
10. `aqi_diff_24h` - 24-hour AQI change

---

## 🧪 Testing

### Run Tests

```bash
# Test data loading
python -c "from app import load_data; print(load_data())"

# Test model loading
python -c "from app import load_models_and_features; print(load_models_and_features())"

# Test predictions
python DIAGNOSE.py
```

### Performance Benchmarks

```bash
# Measure prediction latency
python -c "
import time
from app import load_models_and_features
models, scaler, features, _ = load_models_and_features()
start = time.time()
# [prediction code]
print(f'Latency: {(time.time()-start)*1000:.2f}ms')
"
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Write tests for new features
- Update README for significant changes

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Contact

**Shahzad Hussain**

📧 Email: shahzadhussain9680@gmail.com  
🔗 GitHub: [@Shahzadhussain2005](https://github.com/Shahzadhussain2005)  
💼 LinkedIn: https://www.linkedin.com/in/shahzad-hussain-486a31285?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app

---

## 🙏 Acknowledgments

- **Open-Meteo** for free air quality API
- **AQICN** for real-time AQI data
- **MongoDB Atlas** for cloud database
- **Streamlit** for rapid dashboard development
- **GitHub Actions** for free CI/CD

---

## 📈 Roadmap

- [ ] Add email/SMS alerts for high AQI
- [ ] Implement SHAP explainability
- [ ] Add multi-city support
- [ ] Mobile app development
- [ ] Real-time API endpoint
- [ ] Historical data export
- [ ] Comparison with other cities

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Shahzadhussain2005/Karachi_AQI-MLOps&type=Date)](https://star-history.com/#Shahzadhussain2005/Karachi_AQI-MLOps&Date)

---

<div align="center">

**Made with ❤️ for Karachi**

If you find this project useful, please consider giving it a ⭐!

</div>
