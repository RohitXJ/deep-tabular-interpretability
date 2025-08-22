def ML_model_train(model:object,data:list):
    X_train,_,y_train,_ = data

    model.fit(X_train,y_train)

    return model
