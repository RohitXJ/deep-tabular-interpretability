from sklearn.metrics import classification_report

def ML_model_eval(model:object,test_data:list,type:str):
    if type=="Classification":
        X_test,y_test = test_data
        y_pred = model.predict(X_test)

        print(classification_report(y_true=y_test,y_pred=y_pred))