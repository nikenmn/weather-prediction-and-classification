
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import joblib
from flask import flash

from database_models import db, WeatherData, WeatherCombined, ModelHistory


def preprocess_weather_data():
    try:
        from scipy.interpolate import PchipInterpolator
        from sqlalchemy import text

        raw_data = WeatherData.query.order_by(WeatherData.date).all()
        print(f"🔄 Jumlah data di weather_data_raw: {len(raw_data)}")

        df = pd.DataFrame([{
            'date': row.date,
            'tavg': row.tavg,
            'rh_avg': row.rh_avg,
            'rr': row.rr,
            'ss': row.ss
        } for row in raw_data])

        # 1. Konversi tanggal dan hapus baris error
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Ubah invalid jadi NaT
        df = df.dropna(subset=['date'])  # Hapus baris tanggal invalid
        df = df.sort_values(by='date')



        # 2. Ganti 8888/9999 jadi NaN
        df.replace({8888.0: np.nan, 9999.0: np.nan}, inplace=True)

        # 3. Imputasi ffill + bfill
        print("🧪 NaN sebelum isi:\n", df.isna().sum())
        df = df.ffill()
        df = df.bfill()
        print("✅ NaN sesudah isi:\n", df.isna().sum())

        # 4. Tambah fitur musiman
        df['day_of_year'] = df['date'].dt.dayofyear
        df['year'] = df['date'].dt.year

        # 5. Interpolasi RR per tahun (PCHIP)
        def interpolate_rr(group):
            mask = group['rr'].notna()
            if mask.sum() > 3:
                interpolator = PchipInterpolator(group.loc[mask, 'day_of_year'], group.loc[mask, 'rr'])
                interpolated = interpolator(group['day_of_year'])
                group['rr'] = group['rr'].combine_first(pd.Series(interpolated, index=group.index))
            return group

        df = df.groupby('year', group_keys=False).apply(interpolate_rr)
        df['rr'] = df['rr'].round(1)

        # 6. Simpan ke cleaned
        inserted = 0
        for _, row in df.iterrows():
            tanggal = row['date'].date()
            existing = WeatherCombined.query.filter_by(date=tanggal).first()
            if existing:
                db.session.delete(existing)

            entry = WeatherCombined(
                date=tanggal,
                tavg=row['tavg'],
                rh_avg=row['rh_avg'],
                rr=row['rr'],
                ss=row['ss'],
                day_of_year=row['day_of_year'],
                year=row['year']
            )
            db.session.add(entry)
            inserted += 1

        db.session.commit()
        print(f"✅ Preprocessing selesai. {inserted} baris berhasil dimasukkan ke weather_data_cleaned.")

        # 7. Reset AUTO_INCREMENT ke ID terakhir + 1
        result = db.session.execute(text("SELECT MAX(id) FROM weather_data_cleaned"))
        max_id = result.scalar() or 0
        next_id = max_id + 1
        db.session.execute(text(f"ALTER TABLE weather_data_cleaned AUTO_INCREMENT = {next_id}"))
        db.session.commit()
        print(f"🔁 AUTO_INCREMENT diset ke {next_id}")

        return df

    except Exception as e:
        print(f"🔥 ERROR saat preprocessing: {e}")
        flash(f'Error during preprocessing: {str(e)}', 'danger')
        return pd.DataFrame()





def label_classification_data(df, weights):
    try:
        df_norm = df[['rh_avg', 'tavg', 'rr', 'ss']].copy()
        scaler = MinMaxScaler()
        df_norm[['rh_avg', 'tavg', 'rr', 'ss']] = scaler.fit_transform(df_norm)
        df_norm['rr'] = 1 - df_norm['rr']  # inverse for rainfall

        df['score'] = (
            df_norm['rh_avg'] * weights['rh_avg'] +
            df_norm['tavg'] * weights['tavg'] +
            df_norm['rr'] * weights['rr'] +
            df_norm['ss'] * weights['ss']
        )
        threshold = df['score'].mean()
        df['weather_label'] = df['score'].apply(lambda x: 1 if x > threshold else 0)
        return df
    except Exception as e:
        flash(f'Error during labelling: {str(e)}', 'danger')
        return df


def train_models(df):
    try:
        base_features = ['day_of_year', 'year']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'models/model_{timestamp}'
        os.makedirs(folder_name, exist_ok=True)

        for target, filename in zip(['tavg', 'rh_avg', 'rr', 'ss'],
                                     ['temperature_model.pkl', 'humidity_model.pkl', 'rainfall_model.pkl', 'sunshine_model.pkl']):
            X = df[base_features]
            y = df[target]
            model = ExtraTreesRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            joblib.dump(model, os.path.join(folder_name, filename))

        clf = KNeighborsClassifier(n_neighbors=3)
        features = ['tavg', 'rh_avg', 'rr', 'ss']
        clf.fit(df[features], df['weather_label'])
        joblib.dump(clf, os.path.join(folder_name, 'knn_classifier.pkl'))

        db.session.query(ModelHistory).update({ModelHistory.is_active: False})
        new_entry = ModelHistory(
            folder_path=folder_name,
            data_cutoff=df['date'].max().date(),
            training_date=datetime.now().date(),
            is_active=True
        )
        db.session.add(new_entry)
        db.session.commit()

        flash('Model training successful!', 'success')
        return folder_name

    except Exception as e:
        flash(f'Error during model training: {str(e)}', 'danger')
        return None
