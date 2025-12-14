import joblib
import pandas as pd
from pathlib import Path
import sys

# Add src to path just in case
sys.path.append(str(Path(__file__).resolve().parent / "src"))

try:
    model_path = Path("models/pipeline.joblib")
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Create dummy input based on streamlit app
    inp = pd.DataFrame([{
        "Pclass": 3,
        "Name": "John Doe",
        "Sex": "male",
        "Age": 30,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }])
    
    print("Predicting...")
    pred = model.predict(inp)[0]
    print(f"Prediction: {pred}")

except Exception as e:
    print("An error occurred:")
    import traceback
    traceback.print_exc()
