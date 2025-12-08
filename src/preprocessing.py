
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class TitleExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        def get_title(name):
            m = re.search(r',\s*([^\.]+)\.', str(name))
            return m.group(1).strip() if m else "Unknown"
        X["Title"] = X["Name"].apply(get_title)
        X["Title"] = X["Title"].replace(['Mlle','Ms'],'Miss').replace('Mme','Mrs')
        rare = X["Title"].value_counts()[X["Title"].value_counts()<10].index
        X["Title"] = X["Title"].replace(rare,'Rare')
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        return X

def make_preprocessor():
    numeric = ["Age","Fare","SibSp","Parch","FamilySize"]
    categorical = ["Sex","Embarked","Title","Pclass"]
    num_t = Pipeline([("imp",SimpleImputer(strategy="median")),
                      ("sc",StandardScaler())])
    cat_t = Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                      ("oh",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num",num_t,numeric),
        ("cat",cat_t,categorical)
    ])
