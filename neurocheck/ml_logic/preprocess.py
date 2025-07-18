
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

def preprocess_predict(uploaded_data):
    """
    This function takes a csv passed by the frontend and processes it to be ready for predict.
    """
    # Drop and assign columns
    uploaded_data.columns = uploaded_data.columns.str.strip()

    # Drop class if it's there
    data = uploaded_data.columns.drop('class')

    # Normalize data
    data_norm = np.array(MinMaxScaler().fit_transform(data))

    print(f"Prediction data shape: {data_norm.shape}")

    return data_norm

def preprocess_raw(raw_data):
    """
    This funcitons takes the raw data and applies the following preprocessing steps:
        - dropping and assigning columns
        - creating windows
        - normalizing
        - splitting into training and test set
        """
    # Drop and assign columns
    raw_data.columns = raw_data.columns.str.strip()

    #Load session map which defines start/end rows for each participant-session
    session_map = pd.read_csv('/home/cdennis51/code/Cdennis51/Neurocheck/raw_data/mental_fatigue/Session_Map.csv')

    # Split data into features and target
    features = raw_data.columns.drop('class')
    X= raw_data[features].copy()
    y = raw_data['class'].copy()


    # Creating windows
    sampling_rate = 32
    window_seconds = 10 #Window size = 10 seconds of data--> 320 rows per window
    window_size = sampling_rate * window_seconds
    stride = int(window_size*0.5) #Stride = 50% overlap --> shifts 5 seconds at a time

    #Slide window within each session to avoid mixing subjects/sessions

    X_windows, y_windows, session_ids = [], [], []

    for _, row in session_map.iterrows():     #returns each row of DF as tuple (i,r) _ means ignore index use row
        session_id = row['session_id']
        start = int(row['start_index'])
        end = int(row['end_index'])

        X_session = X.iloc[start:end].values
        y_session = y.iloc[start:end].values

        for i in range(0, len(X_session) - window_size + 1, stride):
            window = X_session[i:i + window_size]
            label = y_session[i + window_size - 1]  # Label taken from the end of the window
            X_windows.append(window)
            y_windows.append(label)
            session_ids.append(session_id)



    # convert to np for modelling
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    session_ids = np.array(session_ids)

    print(len(X))
    print("Window size:", window_size)
    print("Stride:", stride)

    # split into training and test sets using GroupShuffleSplit (grouped by session id)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  #prevents pverlap between train/test split for groups
    train_idx, test_idx = next(splitter.split(X_norm, y_windows, groups=session_ids))

    # Normalize X
    X_norm = np.array([MinMaxScaler().fit_transform(window) for window in X_windows])

    X_train, X_test = X_norm[train_idx], X_norm[test_idx]
    y_train, y_test = y_windows[train_idx], y_windows[test_idx]

    return X_train, X_test, y_train, y_test
