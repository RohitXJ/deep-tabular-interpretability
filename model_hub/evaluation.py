from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score,accuracy_score,mean_absolute_percentage_error
import numpy as np

def ML_model_eval(model: object, test_data: list, type: str):
    X_test,y_test = test_data
    y_pred = model.predict(X_test)

    if type == "Classification":
        print("Classification Report")
        report_str = classification_report(y_true=y_test, y_pred=y_pred)
        report_dict = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
        print(report_str)

        accuracy = report_dict['accuracy']
        f1_score = report_dict['weighted avg']['f1-score']

        summary = "\n--- Model Performance Summary ---\n"
        if accuracy > 0.9 and f1_score > 0.9:
            summary += "Excellent! The model demonstrates high accuracy and a strong F1-score, indicating it is very effective at classifying the data."
        elif accuracy > 0.8 and f1_score > 0.8:
            summary += "Good. The model has a solid accuracy and F1-score, suggesting it performs well for most cases."
        elif accuracy > 0.7 and f1_score > 0.7:
            summary += "Fair. The model's performance is reasonable, but there might be room for improvement. It correctly classifies a moderate amount of the data."
        else:
            summary += "Needs Improvement. The model's accuracy and F1-score are low. It may struggle to make correct predictions and could benefit from further tuning or more data."
        
        summary += f"\n(Accuracy: {accuracy:.2f}, Weighted F1-Score: {f1_score:.2f})"
        print(summary)


    elif type == "Regression":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print("Regression Evaluation")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R Score: {r2:.2f}")
        print(f"Mean Absolute Error Percentage (MAPE): {(mape*100):.2f}%")

        summary = "\n--- Model Performance Summary ---\n"
        if r2 > 0.9 and mape < 0.1:
            summary += "Excellent! The model explains a very high proportion of the variance in the target variable and has a low prediction error."
        elif r2 > 0.8 and mape < 0.2:
            summary += "Good. The model provides a solid fit to the data, explaining a majority of the variance with a reasonable error margin."
        elif r2 > 0.6 and mape < 0.3:
            summary += "Fair. The model has some predictive power but may not be very precise. The R-squared value is moderate, and the error percentage is noticeable."
        else:
            summary += "Needs Improvement. The model does not fit the data well. The R-squared is low (or even negative), and the prediction error is high."

        summary += f"\n(R-squared: {r2:.2f}, MAPE: {mape:.2%})"
        print(summary)

    else:
        raise ValueError("Invalid type. Choose either 'Classification' or 'Regression'.")