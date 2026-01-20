import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.data_loader import load_data
from src.utils.feature_types import get_feature_types
from src.features.preprocessing import build_preprocessing_pipeline
from src.config.config import TARGET_COLUMN
import os



def evaluate_model(y_true, y_pred, model_name):
    rsme = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Evaluation Metrics for {model_name}:")  
    print(f"RMSE: {rsme}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")


def get_feature_names(preprocessor, numerical_features, categorical_features):
    num_features = numerical_features
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(categorical_features)
    return np.concatenate([num_features, cat_features])


ridge_params = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10, 100]
}

lasso_params = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10, 100]
}

elastic_params = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10, 100],
    'regressor__l1_ratio': [0.2, 0.5, 0.8]
}

def main():
    os.makedirs('artifacts', exist_ok=True)
    print("Fetching data from DataBase...")

    df = load_data()

    print("Data Fetched successfully.")

    train_df = df.iloc[:700000]
    val_df = df.iloc[700000:900000]

    # training dataset
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    # validation dataset
    X_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]

    numeric_features, categorical_features = get_feature_types(X_train)

    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    # fro Liner Regression Model
    lr_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    print("Training Linear Regression Model...")
    lr_model.fit(X_train, y_train)

    lr_preds = lr_model.predict(X_val)
    evaluate_model(y_val, lr_preds, "Linear Regression")

    print("Training Ridge Regression Model...")

    ridge_pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    ridge_search = GridSearchCV(ridge_pipeline, ridge_params, cv=3, n_jobs=-1, scoring='r2')
    ridge_search.fit(X_train, y_train)

    ridge_best = ridge_search.best_estimator_
    ridge_preds = ridge_best.predict(X_val)
    print(f"Best Ridge Parameters: {ridge_search.best_params_}")
    evaluate_model(y_val, ridge_preds, "Ridge Regression")



    print("\n Extracting Feature Importances from Ridge Regression Model...")
    ridge_model = ridge_best.named_steps['regressor']
    ridge_preprocessor = ridge_best.named_steps['preprocessor']
    feature_names = get_feature_names(ridge_preprocessor, numeric_features, categorical_features)

    coeefficients = ridge_model.coef_

    feature_importances = (
        pd.DataFrame({
            "feature" : feature_names,
            "importance" : coeefficients,
            "abs_coefficient" : np.abs(coeefficients)
        })
        .sort_values(by="abs_coefficient", ascending=False)
    )

    print(feature_importances.head(15))
    # print("Training Lasso Regression Model...")

    # lasso_pipeline = Pipeline(steps = [
    #     ('preprocessor', preprocessor),
    #     ('regressor', Lasso())
    # ])

    # lasso_search = GridSearchCV(lasso_pipeline, lasso_params, cv=3, n_jobs=-1, scoring='r2')
    # lasso_search.fit(X_train, y_train)
    # lasso_best = lasso_search.best_estimator_
    # lasso_preds = lasso_best.predict(X_val)
    # print(f"Best Lasso Parameters: {lasso_search.best_params_}")
    # evaluate_model(y_val, lasso_preds, "Lasso Regression")




    # for Gradient Boosting Model
    # gb_model = Pipeline(steps=[
    #     ('preprocessor', preprocessor),
    #     ('regressor', GradientBoostingRegressor(
    #         n_estimators=100,
    #         learning_rate=0.1,
    #         max_depth=3,
    #         random_state=42
    #     ))
    # ])

    # print("Training Gradient Boosting Model...")
    # gb_model.fit(X_train, y_train)
    # gb_preds = gb_model.predict(X_val)
    # evaluate_model(y_val, gb_preds, "Gradient Boosting Classifier")

    # Save the best model

    feature_importances.to_csv('artifacts/feature_importances.csv', index=False)
    joblib.dump(ridge_best, 'artifacts/ridge_regression_model.pkl')
    print("Model and feature importances saved to artifacts/ directory.")

if __name__ == "__main__":
    main()

