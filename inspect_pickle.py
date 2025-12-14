import joblib
from pathlib import Path

try:
    path = Path("models/pipeline.joblib")
    print(f"Loading {path}...")
    pipe = joblib.load(path)
    
    # Drill down to SimpleImputer
    # Structure: Pipeline -> ColumnTransformer (prep) -> Pipeline (num_t/cat_t) -> SimpleImputer (imp)
    
    prep = pipe.named_steps['prep']
    num_pipe = prep.named_transformers_['num']
    imputer = num_pipe.named_steps['imp']
    
    print("Inspecting SimpleImputer inside pipe...")
    print(f"Has _fill_dtype: {hasattr(imputer, '_fill_dtype')}")
    if hasattr(imputer, '_fill_dtype'):
        print(f"_fill_dtype value: {imputer._fill_dtype}")
        
except Exception as e:
    print(e)
