

from neurocheck.ml_logic.preprocess import preprocess_predict


def preprocess(frontend_data):
    """
    Takes user input and preprocesses it for prediction.
    Returns a dataframe with one row having one value per feature.
    """

    preprocessed_prediction_data = preprocess_predict(frontend_data)

    print("✅ preprocess() done \n")

    return preprocessed_prediction_data

if __name__ == '__main__':
    preprocess()
