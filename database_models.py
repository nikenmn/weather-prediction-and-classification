
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class WeatherData(db.Model):
    __tablename__ = 'weather_data_raw'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    tavg = db.Column(db.Float)
    rh_avg = db.Column(db.Float)
    rr = db.Column(db.Float)
    ss = db.Column(db.Float)

class WeatherCombined(db.Model):
    __tablename__ = 'weather_data_cleaned'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    year = db.Column(db.Integer)
    day_of_year = db.Column(db.Integer)
    tavg = db.Column(db.Float)
    rh_avg = db.Column(db.Float)
    rr = db.Column(db.Float)
    ss = db.Column(db.Float)

class ModelRegression(db.Model):
    __tablename__ = 'model_regression'
    id = db.Column(db.Integer, primary_key=True)
    parameter_name = db.Column(db.String(10))
    model_path = db.Column(db.Text)
    trained_on = db.Column(db.DateTime)
    version = db.Column(db.Integer)
    model_score_r2 = db.Column(db.Float)
    model_score_rmse = db.Column(db.Float)
    params_json = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=False)

class ModelClassification(db.Model):
    __tablename__ = 'model_classification'
    id = db.Column(db.Integer, primary_key=True)
    model_path = db.Column(db.Text)
    trained_on = db.Column(db.DateTime)
    version = db.Column(db.Integer)
    weights_json = db.Column(db.Text)
    threshold = db.Column(db.Float)
    score_accuracy = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=False)

class ForecastResults(db.Model):
    __tablename__ = 'forecast_results'
    id = db.Column(db.Integer, primary_key=True)
    forecast_date = db.Column(db.Date, nullable=False)
    pred_tavg = db.Column(db.Float)
    pred_rh_avg = db.Column(db.Float)
    pred_rr = db.Column(db.Float)
    pred_ss = db.Column(db.Float)
    classification_label = db.Column(db.Enum('Baik', 'Buruk'))
    classification_reason = db.Column(db.Text)
    generated_at = db.Column(db.DateTime, default=datetime.now)

class TrainingLog(db.Model):
    __tablename__ = 'training_log'
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.Enum('regression', 'classification'))
    parameter_name = db.Column(db.String(10))
    trained_by = db.Column(db.String(100))
    description = db.Column(db.Text)
    trained_at = db.Column(db.DateTime, default=datetime.now)

class ModelHistory(db.Model):
    __tablename__ = 'model_history'
    id = db.Column(db.Integer, primary_key=True)
    folder_path = db.Column(db.String(255), nullable=False)
    data_cutoff = db.Column(db.Date, nullable=False)
    training_date = db.Column(db.Date, nullable=False)
    is_active = db.Column(db.Boolean, default=False)
