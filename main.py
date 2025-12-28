from src.predictor import BodyFatPredictor


import os
import pandas as pd
from src.preprocessing import preprocess_data



if not os.path.exists('data/bodyfat_processed.csv'):
    print("🛠️ Clean data not found. Running preprocessing factory...")
    raw_data = pd.read_csv('data/bodyfat.csv')
    cleaned_data = preprocess_data(raw_data)
    cleaned_data.to_csv('data/bodyfat_processed.csv', index=False)
else:
    print("✅ Clean data already exists. Skipping preprocessing.")




def main():
    print("\n--- 🏋️ BodyFat Professional Estimator ---")
    
    
    try:
        predictor = BodyFatPredictor()
        print("🧠 Engine loaded. Ready for input.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # get User Input Interactively
    print("\n⌨️  Enter measurements (Metric: kg/cm):")
    user_data = {}
    try:
        for feature in predictor.features:
            val = input(f"   Enter {feature}: ")
            user_data[feature] = float(val)
    except ValueError:
        print("❌ Please enter numbers only.")
        return

    # 3. Predict
    try:
        result = predictor.predict(user_data)
        print(f"\n👉 Predicted Body Fat: {result}%")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

    if result < 13:
     print("📈 Status: Athletic")
    elif result < 20:
     print("📈 Status: Fitness / Lean")
    else:
     print("📈 Status: Average / Above Average")




if __name__ == "__main__":
    main()
