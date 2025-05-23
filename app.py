from flask import Flask, render_template, request, Response, send_file, redirect, url_for, flash
from database_models import db, WeatherData, WeatherCombined, ModelRegression, ModelClassification, ForecastResults, TrainingLog, ModelHistory
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import csv
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import os
import joblib
from pathlib import Path
from train_utills import preprocess_weather_data, penjelas_parameter
from sqlalchemy import func





try:
    locale.setlocale(locale.LC_TIME, 'ind')  
except:
    pass  # Amanin kalau locale gak tersedia




app = Flask(__name__)
app.secret_key = 'rahasia123'  

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/jaga_jeruk'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)


ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model_folder_path():
    model_aktif = ModelHistory.query.filter_by(is_active=True).first()
    return model_aktif.folder_path if model_aktif else None


@app.route('/test-db')
def test_db():
    try:
        count = WeatherData.query.count()
        return f"DB Connected! WeatherData rows: {count}"
    except Exception as e:
        return f"DB Connection error: {str(e)}"
    

def safe_load_model(path):
    import joblib
    try:
        model = joblib.load(path)
        return model  # bisa dict (KNN) atau langsung model
    except Exception as e:
        print(f"Gagal load model {path.name}: {e}")
        return None
    

def generate_7day_prediction():
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime, timedelta
    from pathlib import Path
    import joblib
    from database_models import WeatherCombined

    folder_path = Path(get_model_folder_path())
    if not folder_path.exists():
        print("Folder model tidak ditemukan.")
        return []

    try:
        model_rh = joblib.load(folder_path / 'humidity_model.pkl')
        model_tavg = joblib.load(folder_path / 'temperature_model.pkl')
        model_rr = joblib.load(folder_path / 'rainfall_model.pkl')
        model_ss = joblib.load(folder_path / 'sunshine_model.pkl')
        model_classifier = joblib.load(folder_path / 'knn_classifier.pkl')
        print(f"Model aktif digunakan dari folder: {folder_path}")

    except FileNotFoundError as e:
        print(f"File model tidak ditemukan: {e}")
        return []
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        return []


    df_db = WeatherCombined.query.order_by(WeatherCombined.date.asc()).all()
    df = pd.DataFrame([{
        'date': row.date,
        'RH_AVG': row.rh_avg,
        'TAVG': row.tavg,
        'RR': row.rr,
        'SS': row.ss,
        'DAY_OF_YEAR': row.day_of_year,
        'YEAR': row.year
    } for row in df_db])

    for col in ['RH_AVG', 'TAVG', 'RR', 'SS']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['RH_AVG', 'TAVG', 'RR', 'SS'], inplace=True)
    df = df.sort_values(by='date').reset_index(drop=True)

    hasil = []

    hari_dict = {
    0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
    4: "Jumat", 5: "Sabtu", 6: "Minggu"
    }

    for i in range(7):
        tanggal = df['date'].max() + timedelta(days=1)
        day_of_year = tanggal.timetuple().tm_yday
        year = tanggal.year

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        # Fitur RH
        df['RH_AVG_ema_7'] = df['RH_AVG'].ewm(span=7).mean()
        df['RH_AVG_ema_14'] = df['RH_AVG'].ewm(span=14).mean()
        df['RH_AVG_ema_30'] = df['RH_AVG'].ewm(span=30).mean()
        df['RH_AVG_seasonal_mean'] = df['RH_AVG'].rolling(30, min_periods=1).mean()
        df['RH_AVG_lag_1'] = df['RH_AVG'].shift(1)
        df['RH_AVG_lag_3'] = df['RH_AVG'].shift(3)
        df['RH_AVG_lag_7'] = df['RH_AVG'].shift(7)
        df['RH_AVG_lag_14'] = df['RH_AVG'].shift(14)
        df['RH_AVG_lag_30'] = df['RH_AVG'].shift(30)

        X_rh = pd.DataFrame([{
            'RH_AVG_lag_1': df['RH_AVG_lag_1'].iloc[-1],
            'RH_AVG_lag_3': df['RH_AVG_lag_3'].iloc[-1],
            'RH_AVG_lag_7': df['RH_AVG_lag_7'].iloc[-1],
            'RH_AVG_lag_14': df['RH_AVG_lag_14'].iloc[-1],
            'RH_AVG_lag_30': df['RH_AVG_lag_30'].iloc[-1],
            'RH_AVG_ema_7': df['RH_AVG_ema_7'].iloc[-1],
            'RH_AVG_ema_14': df['RH_AVG_ema_14'].iloc[-1],
            'RH_AVG_ema_30': df['RH_AVG_ema_30'].iloc[-1],
            'RH_AVG_seasonal_mean': df['RH_AVG_seasonal_mean'].iloc[-1],
            'day_sin': np.sin(2 * np.pi * day_of_year / 365),
            'day_cos': np.cos(2 * np.pi * day_of_year / 365),
            'YEAR': year
        }])
        pred_rh = model_rh.predict(X_rh)[0]

        # TAVG
        df['TAVG_ema_7'] = df['TAVG'].ewm(span=7).mean()
        df['TAVG_ema_14'] = df['TAVG'].ewm(span=14).mean()
        df['TAVG_ema_30'] = df['TAVG'].ewm(span=30).mean()
        df['TAVG_seasonal_mean'] = df.groupby('DAY_OF_YEAR')['TAVG'].transform('mean')
        df['TAVG_lag_1'] = df['TAVG'].shift(1)
        df['TAVG_lag_3'] = df['TAVG'].shift(3)
        df['TAVG_lag_7'] = df['TAVG'].shift(7)
        df['TAVG_lag_14'] = df['TAVG'].shift(14)
        df['TAVG_lag_30'] = df['TAVG'].shift(30)

        X_tavg = pd.DataFrame([{
            'TAVG_lag_1': df['TAVG_lag_1'].iloc[-1],
            'TAVG_lag_3': df['TAVG_lag_3'].iloc[-1],
            'TAVG_lag_7': df['TAVG_lag_7'].iloc[-1],
            'TAVG_lag_14': df['TAVG_lag_14'].iloc[-1],
            'TAVG_lag_30': df['TAVG_lag_30'].iloc[-1],
            'TAVG_ema_7': df['TAVG_ema_7'].iloc[-1],
            'TAVG_ema_14': df['TAVG_ema_14'].iloc[-1],
            'TAVG_ema_30': df['TAVG_ema_30'].iloc[-1],
            'TAVG_seasonal_mean': df['TAVG_seasonal_mean'].iloc[-1],
            'day_sin': np.sin(2 * np.pi * day_of_year / 365),
            'day_cos': np.cos(2 * np.pi * day_of_year / 365),
            'YEAR': year
        }])
        pred_tavg = model_tavg.predict(X_tavg)[0]

        # RR
        df['RR_lag_1'] = df['RR'].shift(1)
        df['RR_lag_3'] = df['RR'].shift(3)
        df['RR_lag_7'] = df['RR'].shift(7)
        df['RR_lag_14'] = df['RR'].shift(14)
        df['RR_lag_30'] = df['RR'].shift(30)
        df['RR_ema_7'] = df['RR'].ewm(span=7).mean()
        df['RR_ema_14'] = df['RR'].ewm(span=14).mean()
        df['RR_ema_30'] = df['RR'].ewm(span=30).mean()
        df['RR_seasonal_mean'] = df.groupby('DAY_OF_YEAR')['RR'].transform('mean')
        df['is_rain'] = (df['RR'] > 0).astype(int)

        X_rr = pd.DataFrame([{
            'RR_lag_1': df['RR_lag_1'].iloc[-1],
            'RR_lag_3': df['RR_lag_3'].iloc[-1],
            'RR_lag_7': df['RR_lag_7'].iloc[-1],
            'RR_lag_14': df['RR_lag_14'].iloc[-1],
            'RR_lag_30': df['RR_lag_30'].iloc[-1],
            'RR_ema_7': df['RR_ema_7'].iloc[-1],
            'RR_ema_14': df['RR_ema_14'].iloc[-1],
            'RR_ema_30': df['RR_ema_30'].iloc[-1],
            'RR_seasonal_mean': df['RR_seasonal_mean'].iloc[-1],
            'is_rain': df['is_rain'].iloc[-1],
            'day_sin': np.sin(2 * np.pi * day_of_year / 365),
            'day_cos': np.cos(2 * np.pi * day_of_year / 365),
            'YEAR': year
        }])
        pred_rr = np.expm1(model_rr.predict(X_rr)[0])

        # SS
        df['SS_lag_1'] = df['SS'].shift(1)
        df['SS_lag_3'] = df['SS'].shift(3)
        df['SS_lag_7'] = df['SS'].shift(7)
        df['SS_lag_14'] = df['SS'].shift(14)
        df['SS_lag_30'] = df['SS'].shift(30)
        df['SS_ema_7'] = df['SS'].ewm(span=7).mean()
        df['SS_ema_14'] = df['SS'].ewm(span=14).mean()
        df['SS_ema_30'] = df['SS'].ewm(span=30).mean()
        df['SS_seasonal_mean'] = df.groupby('DAY_OF_YEAR')['SS'].transform('mean')

        X_ss = pd.DataFrame([{
            'SS_lag_1': df['SS_lag_1'].iloc[-1],
            'SS_lag_3': df['SS_lag_3'].iloc[-1],
            'SS_lag_7': df['SS_lag_7'].iloc[-1],
            'SS_lag_14': df['SS_lag_14'].iloc[-1],
            'SS_lag_30': df['SS_lag_30'].iloc[-1],
            'SS_ema_7': df['SS_ema_7'].iloc[-1],
            'SS_ema_14': df['SS_ema_14'].iloc[-1],
            'SS_ema_30': df['SS_ema_30'].iloc[-1],
            'SS_seasonal_mean': df['SS_seasonal_mean'].iloc[-1],
            'day_sin': np.sin(2 * np.pi * day_of_year / 365),
            'day_cos': np.cos(2 * np.pi * day_of_year / 365),
            'YEAR': year
        }])
        pred_ss = model_ss.predict(X_ss)[0]

        # Klasifikasi
        fitur_klasifikasi = [[pred_rh, pred_tavg, pred_rr, pred_ss]]
        try:
            label = model_classifier['predict_function'](
                model_classifier['X_train'],
                model_classifier['y_train'],
                fitur_klasifikasi,
                model_classifier['k'],
                model_classifier['weights']
            )[0]
        except:
            label = 'Buruk'
        klasifikasi = 'Baik' if label == 'Baik' or label == 1 else 'Buruk'
        penjelasan = penjelas_parameter(pred_rh, pred_tavg, pred_rr, pred_ss)


        hasil.append({
            'hari': hari_dict[tanggal.weekday()],
            'tanggal': tanggal.strftime('%d/%m'),
            'RH': round(pred_rh, 1),
            'TAVG': round(pred_tavg, 1),
            'RR': round(pred_rr, 1),
            'SS': round(pred_ss, 1),
            'klasifikasi': klasifikasi,
            'penjelasan': penjelas_parameter(pred_rh, pred_tavg, pred_rr, pred_ss),
            'icon': 'goodclassicon.png' if klasifikasi == 'Baik' else 'badclassicon.png',
            'is_rh_warning': pred_rh < 65 or pred_rh > 85,
            'is_tavg_warning': pred_tavg < 25 or pred_tavg > 30,
            'is_rr_warning': pred_rr < 3 or pred_rr > 8,
            'is_ss_warning': pred_ss < 4 or pred_ss > 8,
        })

        df.loc[len(df)] = {
            'date': tanggal,
            'RH_AVG': pred_rh,
            'TAVG': pred_tavg,
            'RR': pred_rr,
            'SS': pred_ss,
            'DAY_OF_YEAR': day_of_year,
            'YEAR': year
        }

    return hasil



    


@app.route('/')
def index():
    hasil = generate_7day_prediction()
    cuaca_hari_ini = hasil[0] if hasil else {}

    if cuaca_hari_ini:
        cuaca_hari_ini['penjelasan'] = penjelas_parameter(
            cuaca_hari_ini['RH'],
            cuaca_hari_ini['TAVG'],
            cuaca_hari_ini['RR'],
            cuaca_hari_ini['SS']
        )

    return render_template("index.html", cuaca_hari_ini=cuaca_hari_ini, cuaca_harian=hasil)



@app.route('/input', methods=['GET'])
def input_data():
    from sqlalchemy import func

    last_date = db.session.query(func.max(WeatherData.date)).scalar()
    today = datetime.now().date()

    if last_date and last_date < today:
        tanggal_dibutuhkan = pd.date_range(start=last_date + timedelta(days=1), end=today, freq='D')
        info_gap = {
            'mulai': tanggal_dibutuhkan[0].strftime('%d-%m-%Y'),
            'akhir': tanggal_dibutuhkan[-1].strftime('%d-%m-%Y'),
            'jumlah': len(tanggal_dibutuhkan)
        }
    else:
        info_gap = None

    return render_template('input.html', current_page='input', last_date=last_date, info_gap=info_gap)


@app.route('/proses_input', methods=['POST'])
def proses_input():
    from sqlalchemy import func

    last_date = db.session.query(func.max(WeatherData.date)).scalar()
    today = datetime.now().date()

    if last_date and last_date < today:
        tanggal_dibutuhkan = pd.date_range(start=last_date + timedelta(days=1), end=today, freq='D')
        info_gap = {
            'mulai': tanggal_dibutuhkan[0].strftime('%d-%m-%Y'),
            'akhir': tanggal_dibutuhkan[-1].strftime('%d-%m-%Y'),
            'jumlah': len(tanggal_dibutuhkan)
        }
    else:
        info_gap = None

    if not all([request.form['TAVG'], request.form['RH'], request.form['RR'], request.form['SS'], request.form['tanggal']]):
        return render_template('input.html', error_kosong=True, current_page='input', last_date=last_date, info_gap=info_gap)

    tanggal_obj = datetime.strptime(request.form['tanggal'], '%Y-%m-%d')
    tanggal = tanggal_obj.date()

    existing = WeatherData.query.filter_by(date=tanggal).first()
    if existing:
        flash("Data untuk tanggal tersebut sudah ada di database.", "warning")
        return render_template('input.html', current_page='input', last_date=last_date, info_gap=info_gap)

    data = WeatherData(
        date=tanggal,
        tavg=float(request.form['TAVG']),
        rh_avg=float(request.form['RH']),
        rr=float(request.form['RR']),
        ss=float(request.form['SS']),
    )

    db.session.add(data)
    db.session.commit()
    return render_template('input.html', sukses=True, current_page='input', last_date=last_date, info_gap=info_gap)


@app.route('/unggah-csv', methods=['GET'])
def upload_csv():
    from sqlalchemy import func

    last_date = db.session.query(func.max(WeatherData.date)).scalar()
    today = datetime.now().date()

    if last_date and last_date < today:
        tanggal_dibutuhkan = pd.date_range(start=last_date + timedelta(days=1), end=today, freq='D')
        info_gap = {
            'mulai': tanggal_dibutuhkan[0].strftime('%d-%m-%Y'),
            'akhir': tanggal_dibutuhkan[-1].strftime('%d-%m-%Y'),
            'jumlah': len(tanggal_dibutuhkan)
        }
    else:
        info_gap = None

    return render_template("upload.html", sukses=False, current_page='upload', last_date=last_date, info_gap=info_gap)


@app.route('/proses_upload_csv', methods=['POST'])
def proses_upload_csv():
    from sqlalchemy import func

    last_date = db.session.query(func.max(WeatherData.date)).scalar()
    today = datetime.now().date()

    if last_date and last_date < today:
        tanggal_dibutuhkan = pd.date_range(start=last_date + timedelta(days=1), end=today, freq='D')
        info_gap = {
            'mulai': tanggal_dibutuhkan[0].strftime('%d-%m-%Y'),
            'akhir': tanggal_dibutuhkan[-1].strftime('%d-%m-%Y'),
            'jumlah': len(tanggal_dibutuhkan)
        }
    else:
        info_gap = None

    if 'file' not in request.files:
        flash('Tidak ada file yang diunggah.')
        return redirect(url_for('upload_csv'))

    file = request.files['file']
    if file.filename == '':
        flash('Nama file kosong.')
        return redirect(url_for('upload_csv'))

    if not allowed_file(file.filename):
        flash('Hanya file CSV yang diperbolehkan.')
        return redirect(url_for('upload_csv'))

    try:
        import pandas as pd
        df = pd.read_csv(file)
        df.rename(columns={'TANGGAL': 'Tanggal', 'RH_AVG': 'RH'}, inplace=True)

        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Tanggal'])
        df = df.sort_values(by='Tanggal')

        # Cek data kosong
        if df[['Tanggal', 'RH', 'TAVG', 'RR', 'SS']].isnull().any().any():
            return render_template("upload.html", error_kosong=True, last_date=last_date, info_gap=info_gap, current_page='upload')

        # Cek tanggal tidak urut
        if not df['Tanggal'].is_monotonic_increasing:
            return render_template("upload.html", error_tanggal=True, last_date=last_date, info_gap=info_gap, current_page='upload')

        # Cek duplikat tanggal
        existing_dates = {row.date for row in WeatherData.query.with_entities(WeatherData.date).all()}
        df = df[~df['Tanggal'].dt.date.isin(existing_dates)]

        if df.empty:
            flash("Semua tanggal pada file sudah ada di database. Tidak ada data baru ditambahkan.", "warning")
            return render_template("upload.html", sukses=False, last_date=last_date, info_gap=info_gap, current_page='upload')

        for _, row in df.iterrows():
            data_raw = WeatherData(
                date=row['Tanggal'].date(),
                rh_avg=row['RH'],
                tavg=row['TAVG'],
                rr=row['RR'],
                ss=row['SS']
            )
            db.session.add(data_raw)
        db.session.commit()

        from train_utills import preprocess_weather_data
        df_cleaned = preprocess_weather_data()
        print(f"Preprocessing selesai: {len(df_cleaned)} baris masuk ke weather_data_cleaned")

        return render_template("upload.html", sukses=True, last_date=last_date, info_gap=info_gap, current_page='upload')

    except Exception as e:
        print(f"ERROR saat upload: {e}")
        flash(f'Terjadi kesalahan saat membaca file: {str(e)}')
        return redirect(url_for('upload_csv'))


@app.context_processor
def inject_duplikat_check():
    from sqlalchemy import func
    existing_dates = {row.date for row in WeatherData.query.with_entities(WeatherData.date).all()}
    return dict(existing_dates=existing_dates)








# HALAMAN PREDIKSI ()
@app.route('/')
def prediksi():
    cuaca_harian = generate_7day_prediction()
    try:
        if isinstance(d['tanggal'], datetime):
            d['tanggal'] = d['tanggal'].strftime('%d-%m-%Y')
        elif isinstance(d['tanggal'], str) and '-' in d['tanggal']:
            d['tanggal'] = datetime.strptime(d['tanggal'], '%Y-%m-%d').strftime('%d-%m-%Y')
    except:
        pass
    return render_template('lihatCuaca.html', cuaca_harian=cuaca_harian, current_page='prediksi')





# UNDUH CSV
@app.route('/unduh')
def unduh():
    cuaca_harian = get_cuaca_7_hari()
    for d in cuaca_harian:
        try:
            if isinstance(d['tanggal'], datetime):
                d['tanggal'] = d['tanggal'].strftime('%d-%m-%Y')
            elif isinstance(d['tanggal'], str) and '-' in d['tanggal']:
                d['tanggal'] = datetime.strptime(d['tanggal'], '%Y-%m-%d').strftime('%d-%m-%Y')
        except:
            pass

    return render_template('unduh.html', cuaca_harian=cuaca_harian, current_page='unduh')

@app.route('/unduh-cuaca-csv')
def unduh_cuaca_csv():
    cuaca_harian = get_cuaca_7_hari()

    df = pd.DataFrame([{
        'Hari': d['hari'],
        'Tanggal': d['tanggal'],
        'RH (%)': d['RH'],
        'TAVG (°C)': d['TAVG'],
        'RR (mm)': d['RR'],
        'SS (jam)': d['SS'],
        'Klasifikasi': d['klasifikasi'],
        'Penjelasan': '; '.join(d['penjelasan'])
    } for d in cuaca_harian])

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return Response(
        buffer,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=cuaca_7_hari.csv"}
    )

@app.route('/unduh-cuaca-pdf')
def unduh_cuaca_pdf():
    cuaca_harian = get_cuaca_7_hari()

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "Laporan Kondisi Cuaca 7 Hari")
    c.setFont("Helvetica", 10)

    y = height - 70
    for i, data in enumerate(cuaca_harian, start=1):
        penjelasan = '; '.join(data['penjelasan'])
        row = f"{i}. {data['hari']} {data['tanggal']} | RH: {data['RH']}% | TAVG: {data['TAVG']}°C | RR: {data['RR']} mm | SS: {data['SS']} jam | {data['klasifikasi']}"
        c.drawString(40, y, row)
        y -= 15
        c.drawString(60, y, f"➤ {penjelasan}")
        y -= 25
        if y < 100:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True,
                     download_name="cuaca_7_hari.pdf",
                     mimetype='application/pdf')

# Fungsi ini menyatukan logika prediksi agar bisa dipakai oleh /unduh
def get_cuaca_7_hari():
    return generate_7day_prediction()




@app.route('/predict', methods=['POST'])
def predict():
    try:
        RH = float(request.form['RH'])
        TAVG = float(request.form['TAVG'])
        RR = float(request.form['RR'])
        SS = float(request.form['SS'])

        # Simulasi hasil prediksi
        result = 'Baik' if TAVG >= 25 else 'Buruk'
        note = 'Ini hanya tampilan demo. Belum ada model asli.'

        return render_template('index.html', 
                               pred=result,
                               catatan=note,
                               RH=RH, TAVG=TAVG, RR=RR, SS=SS)
    except Exception as e:
        return f"Error: {e}"





@app.route('/lihat-cuaca')
def lihat_kondisi_cuaca():
    cuaca_harian = generate_7day_prediction()  
    return render_template('lihatCuaca.html', cuaca_harian=cuaca_harian, current_page='prediksi')







# HALAMAN LATIH MODEL 
@app.route('/latih-model')
def latih_model():
    from sqlalchemy import func

    # Ambil tanggal terakhir dari weather_data_raw
    last_date = db.session.query(func.max(WeatherData.date)).scalar()
    tanggal_terakhir = last_date.strftime('%d-%m-%Y') if last_date else 'Belum ada data'

    # Ambil semua riwayat model dari DB
    riwayat_model = ModelHistory.query.order_by(ModelHistory.id.desc()).all()

    return render_template(
        'latihModel.html',
        tanggal_terakhir=tanggal_terakhir,
        riwayat_model=riwayat_model,
        current_page='latih_model'
    )


# HALAMAN TERAPKAN MODEL
@app.route('/terapkan_model', methods=['POST'])
def terapkan_model():
    model_id = request.form['model_id']
    # Nonaktifkan semua
    ModelHistory.query.update({ModelHistory.is_active: False})
    # Aktifkan model yang dipilih
    model = ModelHistory.query.get(model_id)
    if model:
        model.is_active = True
        db.session.commit()
        flash("✅ Model berhasil diterapkan!", "success")
    else:
        flash("❌ Model tidak ditemukan!", "danger")
    return redirect(url_for('latih_model'))


# Endpoint untuk latih model baru, buat folder versi baru: v1, v2, v3, ...
# Route: /train_model
@app.route('/train_model', methods=['POST'])
def train_model():
    from train_utills import (
        preprocess_weather_data,
        label_classification_data,
        train_models_with_folder
    )

    weights = {'RH_AVG': 0.4, 'TAVG': 0.3, 'RR': 0.2, 'SS': 0.1}

    df = preprocess_weather_data()
    if df.empty:
        flash("❌ Data kosong atau gagal preprocessing.", "danger")
        return redirect(url_for('latih_model'))

    # Pastikan kolom kapital untuk label klasifikasi
    df = df.rename(columns={
        'tavg': 'TAVG',
        'rh_avg': 'RH_AVG',
        'rr': 'RR',
        'ss': 'SS'
    })

    df = label_classification_data(df, weights)

    # Kembalikan kolom ke lowercase untuk training model
    df = df.rename(columns={
        'TAVG': 'tavg',
        'RH_AVG': 'rh_avg',
        'RR': 'rr',
        'SS': 'ss'
    })

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_path = f'models/model_{timestamp}'

    hasil = train_models_with_folder(df, folder_path)
    if folder_path:
        return redirect(url_for('latih_model', trained=1))
    else:
        return redirect(url_for('latih_model', trained=0))








if __name__ == '__main__':
    app.run(debug=True)
