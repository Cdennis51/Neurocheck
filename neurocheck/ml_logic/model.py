#add imports here

def initialize_model():
    pass

def compile_model():
    pass


def train_model(X_train, model):
    """
    This function takes the preprocessed features and trains a model.
    It then saves the model as a pickle file.
    """
    #fit and train model

    #save model
    pass

def save_model(model):
    pass

def evaluate_model(model, X_test):
    """
    This function takes the trained model and evaluates it on X_test.
    It returns the classification outcome and evaluation metrics precision, recall, accuracy.
    """


    pass

def predict(frontend_data_preprocessed):
    """
    This function takes the preprocessed user data and predicts mental fatigue using our model.
    It returns a prediction and scoring metrics.
    """
    #load model from mlflow

    #test shape of frontend_data_preprocessed

    #predict 'class'

    #return prediction metrics

    pass
