import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import text
from database_models import db, WeatherData, WeatherCombined, ModelHistory


def preprocess_weather_data():
    try:
        raw_data = WeatherData.query.order_by(WeatherData.date).all()
        df = pd.DataFrame([{
            'date': row.date,
            'tavg': row.tavg,
            'rh_avg': row.rh_avg,
            'rr': row.rr,
            'ss': row.ss
        } for row in raw_data])

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).sort_values(by='date')
        df.replace({8888.0: np.nan, 9999.0: np.nan}, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        df['day_of_year'] = df['date'].dt.dayofyear
        df['year'] = df['date'].dt.year

        for _, row in df.iterrows():
            existing = WeatherCombined.query.filter_by(date=row['date'].date()).first()
            if existing:
                db.session.delete(existing)
            entry = WeatherCombined(
                date=row['date'].date(),
                tavg=row['tavg'],
                rh_avg=row['rh_avg'],
                rr=row['rr'],
                ss=row['ss'],
                day_of_year=row['day_of_year'],
                year=row['year']
            )
            db.session.add(entry)

        db.session.commit()

        result = db.session.execute(text("SELECT MAX(id) FROM weather_data_cleaned"))
        max_id = result.scalar() or 0
        db.session.execute(text(f"ALTER TABLE weather_data_cleaned AUTO_INCREMENT = {max_id + 1}"))
        db.session.commit()

        return df

    except Exception as e:
        print(f"🔥 ERROR saat preprocessing: {e}")
        db.session.rollback()
        return pd.DataFrame()


def log_model_retrain(folder_path, data_cutoff):
    ModelHistory.query.update({ModelHistory.is_active: False})
    new_log = ModelHistory(
        folder_path=folder_path,
        data_cutoff=data_cutoff,
        training_date=datetime.now().date(),
        is_active=True
    )
    db.session.add(new_log)
    db.session.commit()


def set_active_model(folder_path):
    ModelHistory.query.update({ModelHistory.is_active: False})
    model = ModelHistory.query.filter_by(folder_path=folder_path).first()
    if model:
        model.is_active = True
        db.session.commit()


def get_model_folder_path():
    model_aktif = ModelHistory.query.filter_by(is_active=True).first()
    return model_aktif.folder_path if model_aktif else None


def train_models_with_folder(df, folder_path):
    model_path = Path(folder_path)

    print("📦 Mulai training model dengan folder:", model_path)

    fitur_tavg = [
        'TAVG_lag_1', 'TAVG_lag_3', 'TAVG_lag_7', 'TAVG_lag_14', 'TAVG_lag_30',
        'TAVG_ema_7', 'TAVG_ema_14', 'TAVG_ema_30',
        'TAVG_seasonal_mean', 'day_sin', 'day_cos', 'YEAR'
    ]
    X = df[fitur_tavg]
    y = df['TAVG']
    model_tavg = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model_tavg.fit(X, y)
    joblib.dump(model_tavg, model_path / 'temperature_model.pkl')

    fitur_rh = [
        'RH_AVG_lag_1', 'RH_AVG_lag_3', 'RH_AVG_lag_7', 'RH_AVG_lag_14', 'RH_AVG_lag_30',
        'RH_AVG_ema_7', 'RH_AVG_ema_14', 'RH_AVG_ema_30',
        'RH_AVG_seasonal_mean', 'day_sin', 'day_cos', 'YEAR'
    ]
    X = df[fitur_rh]
    y = df['RH_AVG']
    model_rh = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model_rh.fit(X, y)
    joblib.dump(model_rh, model_path / 'humidity_model.pkl')

    fitur_rr = [
        'RR_lag_1', 'RR_lag_3', 'RR_lag_7', 'RR_lag_14', 'RR_lag_30',
        'RR_ema_7', 'RR_ema_14', 'RR_ema_30',
        'RR_seasonal_mean', 'is_rain', 'day_sin', 'day_cos', 'YEAR'
    ]
    X = df[fitur_rr]
    y = np.log1p(df['RR'])
    model_rr = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model_rr.fit(X, y)
    joblib.dump(model_rr, model_path / 'rainfall_model.pkl')

    fitur_ss = [
        'SS_lag_1', 'SS_lag_3', 'SS_lag_7', 'SS_lag_14', 'SS_lag_30',
        'SS_ema_7', 'SS_ema_14', 'SS_ema_30',
        'day_sin', 'day_cos', 'YEAR'
    ]
    X = df[fitur_ss]
    y = df['SS']
    model_ss = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model_ss.fit(X, y)
    joblib.dump(model_ss, model_path / 'sunshine_model.pkl')

    print("✅ Model berhasil dilatih dan disimpan.")

    data_cutoff = df['date'].max().date() if 'date' in df.columns else datetime.now().date()
    log_model_retrain(str(model_path), data_cutoff)

    return True


# Re-add this for compatibility
from sklearn.preprocessing import MinMaxScaler

def label_classification_data(df, weights):
    df_norm = df[['rh_avg', 'tavg', 'rr', 'ss']].copy()
    scaler = MinMaxScaler()
    df_norm[['rh_avg', 'tavg', 'rr', 'ss']] = scaler.fit_transform(df_norm)
    df_norm['rr'] = 1 - df_norm['rr']

    df['score'] = (
        df_norm['rh_avg'] * weights['rh_avg'] +
        df_norm['tavg'] * weights['tavg'] +
        df_norm['rr'] * weights['rr'] +
        df_norm['ss'] * weights['ss']
    )
    threshold = df['score'].mean()
    df['weather_label'] = df['score'].apply(lambda x: 1 if x > threshold else 0)
    return df


# Re-add for app.py import
from prediction import generate_7day_prediction
