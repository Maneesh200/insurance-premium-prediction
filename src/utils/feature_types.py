from src.config.config import TARGET_COLUMN
import pandas as pd

def get_feature_types(df: pd.DataFrame):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    if TARGET_COLUMN in numeric_features:
        numeric_features.remove(TARGET_COLUMN)
    if TARGET_COLUMN in categorical_features:
        categorical_features.remove(TARGET_COLUMN)
    
    return numeric_features, categorical_features