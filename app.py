from flask import Flask, render_template, request, Response, send_file, redirect, url_for, flash
import joblib
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
import locale




try:
    locale.setlocale(locale.LC_TIME, 'ind')  
except:
    pass  # Amanin kalau locale gak tersedia



# Load model KNN untuk klasifikasi
# model_knn = joblib.load('model/knn_model.pkl')

def klasifikasi_cuaca(RH, TAVG, RR, SS):
    # Ini logika dummy: klasifikasi = 'Baik' kalau semua parameter di range ideal
    if 65 <= RH <= 85 and 25 <= TAVG <= 30 and 3 <= RR <= 8 and 4 <= SS <= 8:
        return 'Baik'
    return 'Buruk'



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




@app.route('/')
def index():
    start_date = datetime.today()

    hari_dict = {
        0: "Senin", 1: "Selasa", 2: "Rabu", 3: "Kamis",
        4: "Jumat", 5: "Sabtu", 6: "Minggu"
    }

    cuaca_harian = []

    for i in range(7):
        tanggal = start_date + timedelta(days=i)
        hari = hari_dict[tanggal.weekday()]
        RH = 75 + i
        TAVG = round(27 + i * 0.3, 1)
        RR = 5 + i % 4
        SS = 7 - i % 3

        penjelasan = penjelas_parameter(RH, TAVG, RR, SS)
        klasifikasi = 'Baik' if all("terlalu" not in p for p in penjelasan) else 'Buruk'

        cuaca_harian.append({
            'hari': hari,
            'tanggal': tanggal.strftime("%d/%m"),
            'TAVG': TAVG,
            'RH': RH,
            'RR': RR,
            'SS': SS,
            'klasifikasi': klasifikasi,
            'penjelasan': penjelasan,
            'icon': 'goodclassicon.png' if klasifikasi == 'Baik' else 'badclassicon.png',
            'is_rh_warning': RH < 65 or RH > 85,
            'is_tavg_warning': TAVG < 25 or TAVG > 30,
            'is_rr_warning': RR < 3 or RR > 8,
            'is_ss_warning': SS < 4 or SS > 8,
        })

    
    for d in cuaca_harian:
        try:
            if isinstance(d['tanggal'], datetime):
                d['tanggal'] = d['tanggal'].strftime('%d-%m-%Y')
            elif isinstance(d['tanggal'], str) and '-' in d['tanggal']:
                d['tanggal'] = datetime.strptime(d['tanggal'], '%Y-%m-%d').strftime('%d-%m-%Y')
        except:
            pass

    cuaca_hari_ini = cuaca_harian[0]
    
    
    return render_template(
        'index.html',
        current_page='index',
        cuaca_hari_ini=cuaca_hari_ini,
        cuaca_harian=cuaca_harian
    )





@app.route('/input', methods=['GET'])
def input_data():
    return render_template('input.html', current_page='input')

# Proses input
@app.route('/proses_input', methods=['POST'])
def proses_input():
    tanggal_obj = datetime.strptime(request.form['tanggal'], '%Y-%m-%d')
    tanggal = tanggal_obj.date()
    year = tanggal.year
    day_of_year = tanggal.timetuple().tm_yday

    data = WeatherData(
        date=tanggal,
        tavg=float(request.form['TAVG']),
        rh_avg=float(request.form['RH']),
        rr=float(request.form['RR']),
        ss=float(request.form['SS']),
        # year dan day_of_year biarkan kosong/dikosongkan
    )


    db.session.add(data)
    db.session.commit()
    return render_template('input.html', sukses=True, current_page='input')



@app.route('/unggah-csv', methods=['GET'])
def upload_csv():
    return render_template("upload.html", sukses=False, current_page='upload')



# Route untuk proses unggah csv
@app.route('/unggah-csv', methods=['POST'])
def proses_upload_csv():
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
        # 1. Baca file dan normalisasi kolom
        df = pd.read_csv(file)
        df.rename(columns={'TANGGAL': 'Tanggal', 'RH_AVG': 'RH'}, inplace=True)

        # 2. Konversi tanggal fleksibel
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Tanggal'])  # Hapus baris tanggal invalid
        df = df.sort_values(by='Tanggal')

        # 3. Validasi kolom
        required_columns = ['Tanggal', 'RH', 'TAVG', 'RR', 'SS']
        if not all(col in df.columns for col in required_columns):
            flash('Format CSV tidak sesuai. Kolom harus mengandung: Tanggal, RH, TAVG, RR, SS.')
            return redirect(url_for('upload_csv'))

        for col in ['RH', 'TAVG', 'RR', 'SS']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                flash(f'Kolom {col} harus berupa angka/desimal.')
                return redirect(url_for('upload_csv'))

        # 4. Simpan ke weather_data_raw
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
        print("✅ Commit sukses ke weather_data_raw")

        # 5. Jalankan preprocessing
        from train_utills import preprocess_weather_data
        df_cleaned = preprocess_weather_data()
        print(f"🧪 Preprocessing selesai: {len(df_cleaned)} baris masuk ke weather_data_cleaned")

        return render_template("upload.html", sukses=True)

    except Exception as e:
        print(f"🔥 ERROR saat upload: {e}")
        flash(f'Terjadi kesalahan saat membaca file: {str(e)}')
        return redirect(url_for('upload_csv'))














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


def penjelas_parameter(RH, TAVG, RR, SS):
    penjelasan = []

    # RH
    if RH < 65:
        penjelasan.append("Kelembapan udara terlalu rendah")
    elif RH > 85:
        penjelasan.append("Kelembapan udara terlalu tinggi")
    else:
        penjelasan.append("Kelembapan udara optimal")

    # TAVG
    if TAVG < 25:
        penjelasan.append("Suhu udara terlalu dingin")
    elif TAVG > 30:
        penjelasan.append("Suhu udara terlalu panas")
    else:
        penjelasan.append("Suhu udara ideal")

    # RR
    if RR < 3:
        penjelasan.append("Curah hujan terlalu sedikit")
    elif RR > 8:
        penjelasan.append("Curah hujan terlalu banyak")
    else:
        penjelasan.append("Curah hujan cukup")

    # SS
    if SS < 4:
        penjelasan.append("Penyinaran kurang")
    elif SS > 8:
        penjelasan.append("Penyinaran terlalu banyak")
    else:
        penjelasan.append("Penyinaran baik")

    return penjelasan

def get_model_folder_path():
    model_aktif = ModelHistory.query.filter_by(is_active=True).first()
    if model_aktif:
        return model_aktif.folder_path
    return None




from sklearn.preprocessing import StandardScaler  # Jika perlu scaling

def generate_7day_prediction():
    from pathlib import Path
    import joblib

    folder = get_model_folder_path()
    if not folder:
        return []

    folder_path = Path(folder)
    try:
        model_files = {
            'tavg': joblib.load(folder_path / 'temperature_model.pkl'),
            'rh_avg': joblib.load(folder_path / 'humidity_model.pkl'),
            'rr': joblib.load(folder_path / 'rainfall_model.pkl'),
            'ss': joblib.load(folder_path / 'sunshine_model.pkl'),
            'classifier': joblib.load(folder_path / 'knn_classifier.pkl'),
        }
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    today = datetime.now()
    hasil = []

    for i in range(7):
        tanggal = today + timedelta(days=i)
        day_of_year = tanggal.timetuple().tm_yday
        year = tanggal.year
        fitur = [[day_of_year, year]]

        pred_tavg = model_files['tavg'].predict(fitur)[0]
        pred_rh = model_files['rh_avg'].predict(fitur)[0]
        pred_rr = model_files['rr'].predict(fitur)[0]
        pred_ss = model_files['ss'].predict(fitur)[0]

        # Klasifikasi
        fitur_klasifikasi = [[pred_tavg, pred_rh, pred_rr, pred_ss]]
        label = model_files['classifier'].predict(fitur_klasifikasi)[0]
        klasifikasi = 'Baik' if label == 1 else 'Buruk'

        penjelasan = penjelas_parameter(pred_rh, pred_tavg, pred_rr, pred_ss)

        hasil.append({
            'hari': tanggal.strftime('%A'),
            'tanggal': tanggal.strftime('%d-%m-%Y'),
            'TAVG': round(pred_tavg, 1),
            'RH': round(pred_rh, 1),
            'RR': round(pred_rr, 1),
            'SS': round(pred_ss, 1),
            'klasifikasi': klasifikasi,
            'penjelasan': penjelasan,
            'icon': 'goodclassicon.png' if klasifikasi == 'Baik' else 'badclassicon.png',
            'is_rh_warning': pred_rh < 65 or pred_rh > 85,
            'is_tavg_warning': pred_tavg < 25 or pred_tavg > 30,
            'is_rr_warning': pred_rr < 3 or pred_rr > 8,
            'is_ss_warning': pred_ss < 4 or pred_ss > 8,
        })

    return hasil



@app.route('/lihat-cuaca')
def lihat_kondisi_cuaca():
    cuaca_harian = generate_7day_prediction()
    try:
        if isinstance(d['tanggal'], datetime):
            d['tanggal'] = d['tanggal'].strftime('%d-%m-%Y')
        elif isinstance(d['tanggal'], str) and '-' in d['tanggal']:
            d['tanggal'] = datetime.strptime(d['tanggal'], '%Y-%m-%d').strftime('%d-%m-%Y')
    except:
        pass
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




@app.route('/train_model', methods=['POST'])
def train_model():
    from train_utils import preprocess_weather_data, label_classification_data, train_models

    # Bobot Copeland/WSM — pastikan ini sesuai Copeland score final
    weights = {
        'rh_avg': 0.4,
        'tavg': 0.3,
        'rr': 0.2,
        'ss': 0.1
    }

    # 1. Jalankan preprocessing data dari tabel weather_data_raw → weather_data_cleaned
    df = preprocess_weather_data()
    if df.empty:
        flash("❌ Data kosong atau gagal preprocessing.", "danger")
        return redirect(url_for('latih_model'))

    # 2. Tambahkan label klasifikasi berdasarkan WSM
    df = label_classification_data(df, weights)

    # 3. Latih model dan simpan ke folder
    folder_path = train_models(df)

    if folder_path:
        flash("✅ Model baru berhasil dilatih dan diterapkan!", "success")
    else:
        flash("❌ Training gagal, model tidak diterapkan.", "danger")

    return redirect(url_for('latih_model'))






if __name__ == '__main__':
    app.run(debug=True)
