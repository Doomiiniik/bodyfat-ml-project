from src.predictor import BodyFatPredictor

def main():
    print("--- 🏋️ BodyFat Professional Estimator ---")
    
    # 1. Initialize the Engine
    # (It automatically finds the models because we fixed the pathing in predictor.py)
    try:
        predictor = BodyFatPredictor()
        print("✅ System initialized. Model & Scaler loaded.")
    except Exception as e:
        print(f"❌ Critical Error: Could not load models. {e}")
        return

    # 2. Display what features the model actually needs
    # This helps you know what data to provide without guessing!
    required_features = predictor.features
    print(f"\n📋 Model expects these measurements: {required_features}")

    # 3. Define the Input Data
    # NOTE: You only need to provide the features listed above. 
    # Extra keys (like 'Name') are safely ignored by our predictor logic.
    user_data = {
        'Abdomen': 94.0,   # cm
        'Weight': 80.0,   # lbs (check your dataset if it was lbs or kg!)
        'Wrist': 18.5,     # cm
        'Neck': 38.0,      # cm
        'Height': 70.0     # inches
    }

    # 4. Run Prediction
    try:
        result = predictor.predict(user_data)
        print(f"\n👉 Predicted Body Fat: {result}%")
        
        # Simple health interpretation (optional 'Pro' touch)
        if result < 6:
            print("   (Category: Essential Fat / Athlete)")
        elif result < 24:
            print("   (Category: Fitness / Average)")
        else:
            print("   (Category: Obese)")
            
    except KeyError as e:
        print(f"❌ Input Error: You forgot a required feature! Missing: {e}")
    except Exception as e:
        print(f"❌ Prediction Error: {e}")

if __name__ == "__main__":
    main()
