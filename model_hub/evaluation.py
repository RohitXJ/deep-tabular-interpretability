from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score,accuracy_score,mean_absolute_percentage_error
import numpy as np

def ML_model_eval(model: object, test_data: list, type: str):
    X_test, y_test = test_data
    y_pred = model.predict(X_test)

    if type == "Classification":
        print("Classification Report")
        print(classification_report(y_true=y_test, y_pred=y_pred))
        print(accuracy_score(y_true=y_test, y_pred=y_pred))

    elif type == "Regression":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Regression Evaluation")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error Percentage (MAPE): {mape}")

    else:
        raise ValueError("Invalid type. Choose either 'Classification' or 'Regression'.")
