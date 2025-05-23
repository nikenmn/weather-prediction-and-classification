import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import text
from database_models import db, WeatherData, WeatherCombined, ModelHistory
from flask import flash


def get_model_folder_path():
    model_aktif = ModelHistory.query.filter_by(is_active=True).first()
    return model_aktif.folder_path if model_aktif else None


def generate_future_features(parameter: str, days_ahead: int = 7):
    rows = WeatherCombined.query.order_by(WeatherCombined.date).all()
    df = pd.DataFrame([{
        'date': row.date,
        'year': row.year,
        'day_of_year': row.day_of_year,
        'TAVG': row.tavg,
        'RH_AVG': row.rh_avg,
        'RR': row.rr,
        'SS': row.ss
    } for row in rows])

    df = df.sort_values(by='date').copy()

    param = parameter.upper()
    target = 'RH_AVG' if param == 'RH' else param

    for l in [1, 3, 7, 14, 30]:
        df[f'{target}_lag_{l}'] = df[target].shift(l)
    for span in [7, 14, 30]:
        df[f'{target}_ema_{span}'] = df[target].ewm(span=span).mean()

    df[f'{target}_seasonal_mean'] = df.groupby('day_of_year')[target].transform('mean')
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    if param == 'RR':
        df['is_rain'] = (df['RR'] > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    last_row = df.iloc[-1]

    results = []
    for i in range(1, days_ahead + 1):
        future_date = last_row['date'] + timedelta(days=i)
        future_year = future_date.year
        future_dayofyear = future_date.timetuple().tm_yday

        row = {
            'YEAR': future_year,
            'day_sin': np.sin(2 * np.pi * future_dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * future_dayofyear / 365),
            f'{target}_seasonal_mean': df[df['day_of_year'] == future_dayofyear][f'{target}_seasonal_mean'].mean()
        }

        for l in [1, 3, 7, 14, 30]:
            row[f'{target}_lag_{l}'] = df[target].iloc[-l]
        for span in [7, 14, 30]:
            row[f'{target}_ema_{span}'] = df[f'{target}_ema_{span}'].iloc[-1]

        if param == 'RR':
            row['is_rain'] = int(df['RR'].iloc[-1] > 0)

        results.append(row)

        df = pd.concat([
            df,
            pd.DataFrame([{**row, target: 0, 'date': future_date, 'day_of_year': future_dayofyear, 'year': future_year}])
        ], ignore_index=True)

    return pd.DataFrame(results)


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
    

def label_classification_data(df, weights):
    try:
        print("🔍 Mulai proses labeling klasifikasi dengan Copeland + WSM...")

        # Normalisasi nama kolom agar cocok
        rename_map = {
            'tavg': 'TAVG',
            'rh_avg': 'RH_AVG',
            'rr': 'RR',
            'ss': 'SS'
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ideal range untuk pertumbuhan optimal pohon jeruk
        ideal_ranges = {
            'TAVG': (25, 30),
            'RH_AVG': (50, 85),
            'SS': (6, 8),
            'RR': (2.6, 8)
        }

        # Fungsi untuk memberi nilai 1 atau 0 tergantung pada apakah parameter dalam rentang ideal
        def parameter_score(value, min_ideal, max_ideal):
            return 1.0 if min_ideal <= value <= max_ideal else 0.0

        # Hitung nilai WSM untuk setiap parameter
        for param in ['TAVG', 'RH_AVG', 'SS', 'RR']:
            min_ideal, max_ideal = ideal_ranges[param]
            if param in df.columns:
                df[f'{param}_WSM'] = df[param].apply(lambda x: parameter_score(x, min_ideal, max_ideal) * weights[param])
            else:
                raise KeyError(f"Kolom {param} tidak ditemukan di DataFrame!")

        # Total WSM score
        df['WSM_Score'] = df[[f'{param}_WSM' for param in ['TAVG', 'RH_AVG', 'RR', 'SS']]].sum(axis=1)

        # Label: 1 = Baik, 0 = Buruk
        df['weather_label'] = df['WSM_Score'].apply(lambda x: 1 if x >= 0.5 else 0)

        print("✅ Label klasifikasi berhasil ditambahkan.")
        return df

    except Exception as e:
        print("🔥 ERROR di fungsi label_classification_data:", e)
        import traceback
        traceback.print_exc()
        return df





def log_model_retrain(folder_path, data_cutoff):
    try:
        # Nonaktifkan model aktif sebelumnya
        ModelHistory.query.update({ModelHistory.is_active: False})

        # Buat entri baru
        new_log = ModelHistory(
            folder_path=folder_path,
            data_cutoff=data_cutoff,
            training_date=datetime.now().date(),
            is_active=True
        )

        db.session.add(new_log)
        db.session.commit()
        print(f"✅ Model {folder_path} berhasil dicatat sebagai retrain terbaru.")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Gagal mencatat model retrain: {e}")



def penjelas_parameter(RH, TAVG, RR, SS):
    penjelasan = []

    # Kelembapan Udara (RH)
    if RH < 50:
        penjelasan.append("Kelembapan terlalu rendah (berisiko tanaman layu dan kurang efisien menyerap nutrisi)")
    elif RH > 85:
        penjelasan.append("Kelembapan terlalu tinggi (memicu penyakit jamur dan pembusukan buah)")
    else:
        penjelasan.append("Kelembapan udara berada pada tingkat optimal.")

    # Suhu Udara (TAVG)
    if TAVG < 25:
        penjelasan.append("Suhu terlalu rendah (fotosintesis tidak maksimal, pertumbuhan tanaman melambat)")
    elif TAVG > 30:
        penjelasan.append("Suhu terlalu tinggi (meningkatkan penguapan dan menyebabkan stres tanaman)")
    else:
        penjelasan.append("Suhu udara berada dalam kisaran ideal.")

    # Curah Hujan (RR)
    if RR < 2.6:
        penjelasan.append("Curah hujan terlalu rendah (tanaman kekurangan air, tanah bisa menjadi kering)")
    elif RR > 8:
        penjelasan.append("Curah hujan terlalu tinggi (potensi menyebabkan banjir kecil dan akar mudah busuk)")
    else:
        penjelasan.append("Curah hujan mendukung pertumbuhan optimal.")

    # Lama Penyinaran (SS)
    if SS < 6:
        penjelasan.append("Penyinaran kurang (proses fotosintesis tidak optimal)")
    elif SS > 8:
        penjelasan.append("Penyinaran terlalu banyak (suhu sekitar meningkat, tanaman rentan stres)")
    else:
        penjelasan.append("Lama penyinaran sesuai kebutuhan tanaman.")

    return penjelasan


def train_models_with_folder(df, folder_path):
    try:
        import joblib
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np
        import os
        from database_models import ModelHistory

        print("🔧 Mulai training model dengan fitur kompleks...")

        # Siapkan folder model
        os.makedirs(folder_path, exist_ok=True)

        # Tambah fitur musiman
        df['day_of_year'] = df['date'].dt.dayofyear
        df['YEAR'] = df['date'].dt.year
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # RH features
        df['RH_AVG_ema_7'] = df['rh_avg'].ewm(span=7).mean()
        df['RH_AVG_ema_14'] = df['rh_avg'].ewm(span=14).mean()
        df['RH_AVG_ema_30'] = df['rh_avg'].ewm(span=30).mean()
        df['RH_AVG_seasonal_mean'] = df['rh_avg'].rolling(30, min_periods=1).mean()
        for lag in [1, 3, 7, 14, 30]:
            df[f'RH_AVG_lag_{lag}'] = df['rh_avg'].shift(lag)

        # TAVG features
        df['TAVG_ema_7'] = df['tavg'].ewm(span=7).mean()
        df['TAVG_ema_14'] = df['tavg'].ewm(span=14).mean()
        df['TAVG_ema_30'] = df['tavg'].ewm(span=30).mean()
        df['TAVG_seasonal_mean'] = df.groupby('day_of_year')['tavg'].transform('mean')
        for lag in [1, 3, 7, 14, 30]:
            df[f'TAVG_lag_{lag}'] = df['tavg'].shift(lag)

        # RR features
        df['RR_ema_7'] = df['rr'].ewm(span=7).mean()
        df['RR_ema_14'] = df['rr'].ewm(span=14).mean()
        df['RR_ema_30'] = df['rr'].ewm(span=30).mean()
        df['RR_seasonal_mean'] = df.groupby('day_of_year')['rr'].transform('mean')
        df['is_rain'] = (df['rr'] > 0).astype(int)
        for lag in [1, 3, 7, 14, 30]:
            df[f'RR_lag_{lag}'] = df['rr'].shift(lag)

        # SS features
        df['SS_ema_7'] = df['ss'].ewm(span=7).mean()
        df['SS_ema_14'] = df['ss'].ewm(span=14).mean()
        df['SS_ema_30'] = df['ss'].ewm(span=30).mean()
        df['SS_seasonal_mean'] = df.groupby('day_of_year')['ss'].transform('mean')
        for lag in [1, 3, 7, 14, 30]:
            df[f'SS_lag_{lag}'] = df['ss'].shift(lag)

        df = df.dropna().reset_index(drop=True)

        # Definisi fitur input berdasarkan generate_7day_prediction
        features_rh = ['RH_AVG_lag_1', 'RH_AVG_lag_3', 'RH_AVG_lag_7', 'RH_AVG_lag_14', 'RH_AVG_lag_30',
                       'RH_AVG_ema_7', 'RH_AVG_ema_14', 'RH_AVG_ema_30', 'RH_AVG_seasonal_mean',
                       'day_sin', 'day_cos', 'YEAR']
        features_tavg = ['TAVG_lag_1', 'TAVG_lag_3', 'TAVG_lag_7', 'TAVG_lag_14', 'TAVG_lag_30',
                         'TAVG_ema_7', 'TAVG_ema_14', 'TAVG_ema_30', 'TAVG_seasonal_mean',
                         'day_sin', 'day_cos', 'YEAR']
        features_rr = ['RR_lag_1', 'RR_lag_3', 'RR_lag_7', 'RR_lag_14', 'RR_lag_30',
                       'RR_ema_7', 'RR_ema_14', 'RR_ema_30', 'RR_seasonal_mean', 'is_rain',
                       'day_sin', 'day_cos', 'YEAR']
        features_ss = ['SS_lag_1', 'SS_lag_3', 'SS_lag_7', 'SS_lag_14', 'SS_lag_30',
                       'SS_ema_7', 'SS_ema_14', 'SS_ema_30', 'SS_seasonal_mean',
                       'day_sin', 'day_cos', 'YEAR']

        model_rh = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model_rh.fit(df[features_rh], df['rh_avg'])
        joblib.dump(model_rh, os.path.join(folder_path, 'humidity_model.pkl'))

        model_tavg = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model_tavg.fit(df[features_tavg], df['tavg'])
        joblib.dump(model_tavg, os.path.join(folder_path, 'temperature_model.pkl'))

        model_rr = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model_rr.fit(df[features_rr], np.log1p(df['rr']))
        joblib.dump(model_rr, os.path.join(folder_path, 'rainfall_model.pkl'))

        model_ss = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model_ss.fit(df[features_ss], df['ss'])
        joblib.dump(model_ss, os.path.join(folder_path, 'sunshine_model.pkl'))

        # Klasifikasi
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(df[['tavg', 'rh_avg', 'rr', 'ss']], df['weather_label'])
        joblib.dump(clf, os.path.join(folder_path, 'knn_classifier.pkl'))

        # Simpan ke database
        db.session.query(ModelHistory).update({ModelHistory.is_active: False})
        new_entry = ModelHistory(
            folder_path=folder_path,
            data_cutoff=df['date'].max().date(),
            training_date=datetime.now().date(),
            is_active=True
        )
        db.session.add(new_entry)
        db.session.commit()
        print("✅ Semua model dilatih dan disimpan dengan fitur kompleks.")
        return folder_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Gagal training model: {str(e)}", 'danger')
        return None





__all__ = [
    'generate_future_features',
    'preprocess_weather_data',
    'penjelas_parameter',
    'label_classification_data',
    'train_models_with_folder',
    'get_model_folder_path',
    'set_active_model'
]

