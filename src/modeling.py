
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import TitleExtractor, make_preprocessor

def train_and_save():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "train.csv")
    y=df["Survived"]; X=df.drop(["Survived","Ticket","Cabin","PassengerId"],axis=1)
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    pipe=Pipeline([
        ("title",TitleExtractor()),
        ("prep",make_preprocessor()),
        ("model",RandomForestClassifier(n_estimators=200,random_state=42))
    ])
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xv)
    print("Accuracy:",accuracy_score(yv,pred))
    joblib.dump(pipe, str(root / "models" / "pipeline.joblib"))
if __name__=="__main__":
    train_and_save()
