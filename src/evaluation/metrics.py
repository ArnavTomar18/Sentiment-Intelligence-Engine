from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"\n  {model_name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {"accuracy": acc, "f1": f1}

def evaluate_regression(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a regression model.
    Returns rmse so callers can compare models and pick the best.
    """
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"\n  {model_name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    return rmse 