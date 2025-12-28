import joblib
import pandas as pd
import os

class BodyFatPredictor:
    def __init__(self):
        # 1. Get the absolute path to the directory where THIS script lives (the 'src' folder)
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Go up one level to the root, then into 'models'
        models_dir = os.path.join(base_path, '..', 'models')
        
        # 3. Define full paths to the files
        model_path = os.path.join(models_dir, 'bodyfat_model.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        feat_path = os.path.join(models_dir, 'features.pkl')

        # 4. Load everything
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = joblib.load(feat_path)

    def predict(self, input_data):
        df = pd.DataFrame([input_data])
        df = df[self.features] 
        scaled_data = self.scaler.transform(df)
        prediction = self.model.predict(scaled_data)
        return round(prediction[0], 2)