### Agent Instructions
Act like a `dev` and solve the problems listed here or entered by the user, make changes in the code files, do not try to make any changes in the notebook files
For this operation you will be needing to read codes from these two files,@ANN_architecture.py and @SHAP_for_ANNs.py so try not to change anthing here, and most importatly do you work of integrating this DL models in such a way that the existing and working ML models codes shouldnt be disturbed, i reperat , the ML models works shouldnt be hampered, make seprate files for DL part. Now even the DL part is also working so keep a check that any new changes shouldnt disrupt any existing processes.

### Probelems to solve
* In the interpretation page, we can see that after the waterfall graph, the predicted output is being printed for all DL and ML models, thats ok, but for all classification models of ML and DL, we are getting to see log-odd values, which i guess the naive users won't be able to understand, so i want you to add 2 more texts to be shown after that predicted output text that is, 
** The Predicted output (this already exists so let it be)
** The Original Predicted Output (the one we are getting class values after converting it from the log-odd values, do show it in a formula format that how are we getting that value and also the value of the class)
** The Original True Value (the value that was supposed to be the output)

* In the interpretation page, you will be able to see categorical data showing area, so if during the use of any ML or DL classificaiton models, if the target column is seen to have categorical values instead of numbers then make sure to show that in the categorical data showing area too with a separate lable that these are the target values encodede vs non-encoded.

* Can we introduce a live loading screen before model result page and before interpretation page, the loading logo will actually act like a live progress meter where the model training progress will be shown to give it a realastic feel, can you please implement these, shouldnt look any lame work ok, it should look good and realastic with all required texts or messages around that whtas hapenning and all, focus on the css here to make it look good.