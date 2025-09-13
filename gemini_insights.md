### Plan for Integrating Deep Learning Models

1.  **Create a New File for Deep Learning Models:** To keep the code organized, create a new file named `dl_models.py`. This file will contain all the code for initializing and training the deep learning models.

2.  **Implement Model Initialization:** In `dl_models.py`, write functions to initialize each of the models: FNN (MLP), TabNet, TabTransformer, NODE, and FT-Transformer.

3.  **Implement a Training and Evaluation Function:** Create a generic function to handle the training and evaluation of these models. This function will take a model and the dataset as input.

4.  **Integrate with `main.py`:** Modify the existing `main.py` file to call these new functions and integrate the deep learning models into the command-line interface.