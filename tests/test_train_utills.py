import unittest
import pandas as pd
import numpy as np
from train_utills import train_models_with_folder, generate_7day_prediction, preprocess_weather_data, label_classification_data
from pathlib import Path
import os
from app import app  # Import the Flask app for context

class TestTrainUtills(unittest.TestCase):

    def setUp(self):
        # Prepare a small sample dataframe for testing
        data = {
            'date': pd.date_range(start='2023-01-01', periods=40, freq='D'),
            'tavg': np.random.uniform(20, 30, 40),
            'rh_avg': np.random.uniform(60, 90, 40),
            'rr': np.random.uniform(0, 10, 40),
            'ss': np.random.uniform(0, 10, 40),
            'day_of_year': pd.date_range(start='2023-01-01', periods=40, freq='D').dayofyear,
            'year': pd.date_range(start='2023-01-01', periods=40, freq='D').year
        }
        self.df = pd.DataFrame(data)
        weights = {
            'rh_avg': 0.4,
            'tavg': 0.3,
            'rr': 0.2,
            'ss': 0.1
        }
        self.df = label_classification_data(self.df, weights)
        self.test_folder = Path('tests/test_models')
        if self.test_folder.exists():
            for f in self.test_folder.iterdir():
                f.unlink()
        else:
            os.makedirs(self.test_folder)

    def test_train_models_with_folder(self):
        folder_path = str(self.test_folder)
        result = train_models_with_folder(self.df, folder_path)
        self.assertEqual(result, folder_path)
        # Check if model files are created
        expected_files = ['temperature_model.pkl', 'humidity_model.pkl', 'rainfall_model.pkl', 'sunshine_model.pkl', 'knn_classifier.pkl']
        for filename in expected_files:
            self.assertTrue((self.test_folder / filename).exists())

    def test_generate_7day_prediction(self):
        with app.app_context():
            # This test assumes models are already trained and saved in test_folder
            # We will train models first
            train_models_with_folder(self.df, str(self.test_folder))
            # Mock get_model_folder_path to return test_folder path
            import train_utills
            original_get_model_folder_path = train_utills.get_model_folder_path
            train_utills.get_model_folder_path = lambda: str(self.test_folder)
            try:
                predictions = generate_7day_prediction()
                self.assertIsInstance(predictions, list)
                self.assertEqual(len(predictions), 7)
                for day_pred in predictions:
                    self.assertIn('TAVG', day_pred)
                    self.assertIn('RH', day_pred)
                    self.assertIn('RR', day_pred)
                    self.assertIn('SS', day_pred)
                    self.assertIn('klasifikasi', day_pred)
            finally:
                train_utills.get_model_folder_path = original_get_model_folder_path

if __name__ == '__main__':
    unittest.main()
