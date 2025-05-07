from flask import Flask, render_template, request, Response, send_file, redirect, url_for, flash
import joblib
from flask_sqlalchemy import SQLAlchemy
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

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/cuaca_jeruk'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Model tabel sesuai DB 
class WeatherData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.Date, nullable=False)
    RH_AVG = db.Column(db.Float, nullable=False)
    TAVG = db.Column(db.Float, nullable=False)
    RR = db.Column(db.Float, nullable=False)
    SS = db.Column(db.Float, nullable=False)


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


@app.route('/proses_input', methods=['POST'])
def proses_input():
    tanggal = datetime.strptime(request.form['tanggal'], '%Y-%m-%d').date()
    data = WeatherData(
        tanggal=tanggal,
        RH_AVG=float(request.form['RH']),
        TAVG=float(request.form['TAVG']),
        RR=float(request.form['RR']),
        SS=float(request.form['SS']),
    )
    db.session.add(data)
    db.session.commit()
    return render_template('input.html', sukses=True, current_page='input')



# Route untuk form unggah csv
@app.route('/unggah-csv')
def upload_csv():
    return render_template("upload.html", sukses=False, current_page='input')





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
        df = pd.read_csv(file)

        # Cek kolom
        required_columns = ['Tanggal', 'RH', 'TAVG', 'RR', 'SS']
        if not all(col in df.columns for col in required_columns):
            flash('Format CSV tidak sesuai. Kolom harus mengandung: Tanggal, RH, TAVG, RR, SS.')
            return redirect(url_for('upload_csv'))

        # Cek tipe data numerik
        for col in ['RH', 'TAVG', 'RR', 'SS']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                flash(f'Kolom {col} harus berupa angka/desimal.')
                return redirect(url_for('upload_csv'))

        # Jika semua valid → tampilkan popup
        return render_template("upload.html", sukses=True)

    except Exception as e:
        flash(f'Terjadi kesalahan saat membaca file: {str(e)}')
        return redirect(url_for('upload_csv'))







# HALAMAN PREDIKSI (nanti bisa isi dummy dulu)
@app.route('/')
def prediksi():
    # Simulasikan data cuaca_harian sementara
    dummy_data = [{
        'hari': 'Kamis', 'tanggal': '2025-05-08', 'RH': 75, 'TAVG': 27.0, 'RR': 5.0, 'SS': 7.0,
        'klasifikasi': 'Baik', 'penjelasan': ['Suhu cukup baik', 'Kelembapan optimal']
    }]
    return render_template('index.html', cuaca_harian=dummy_data, current_page='prediksi')



# UNDUH CSV
@app.route('/unduh')
def unduh():
    cuaca_harian = get_cuaca_7_hari()
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



from sklearn.preprocessing import StandardScaler  # Jika perlu scaling

def generate_7day_prediction():
    today = datetime.now(pytz.timezone("Asia/Jakarta"))
    hasil = []

    for i in range(7):
        tgl = today + timedelta(days=i)
        RH = 75 + i
        TAVG = 25 + (i % 3)
        RR = 5 + i
        SS = 7 - (i % 3)

        penjelasan = penjelas_parameter(RH, TAVG, RR, SS)
        klasifikasi = klasifikasi_cuaca(RH, TAVG, RR, SS)  # pakai model!

        hasil.append({
            'hari': tgl.strftime('%A'),
            'tanggal': tgl.strftime('%d/%m/%y'),
            'RH': RH,
            'TAVG': TAVG,
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

    return hasil

@app.route('/lihat-cuaca')
def lihat_kondisi_cuaca():
    cuaca_harian = generate_7day_prediction()
    return render_template('lihatCuaca.html', cuaca_harian=cuaca_harian, current_page='prediksi')






if __name__ == '__main__':
    app.run(debug=True)
