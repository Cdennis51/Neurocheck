

from neurocheck.ml_logic.preprocess import preprocess_predict
from neurocheck.ml_logic.model import predict


def preprocess(frontend_data):
    """
    Takes user input and preprocesses it for prediction.
    Returns a dataframe with one row having one value per feature.
    """

    preprocessed_prediction_data = preprocess_predict(frontend_data)

    print("✅ preprocess() done \n")

    return preprocessed_prediction_data

def predict():
    """
    Takes the preprocessed user data and makes a prediciton using the current production model.
    Returns the predicted class.
    """
    pass

def evaluate():
    """
    Evaluates the prediciton? Not sure if we'll have the actual correct class?
    """

    pass

if __name__ == '__main__':
    preprocess()
    predict()
    #evaluate()
    
