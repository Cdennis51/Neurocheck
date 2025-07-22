
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit

# This can also be deleted, the function preprocess_eeg_data takes this into account.
def preprocess_predict(frontend_data):
    """
    This function takes a csv passed by the frontend and processes it to be ready for predict.
    """
    # Drop and assign columns
    frontend_data.columns = frontend_data.columns.str.strip()

    # Drop class if it's there
    data = frontend_data.columns.drop('class')

    # Normalize data
    data_norm = np.array(MinMaxScaler().fit_transform(data))

    print(f"Prediction data shape: {data_norm.shape}")

    return data_norm

def preprocess_eeg_data(csv_path):
    """
    This takes raw CSV file from one of the sessions either morning or night and scales - preparing the data for XGBoost.
    """
    # Load the CSV
    raw_eeg = pd.read_csv(csv_path)

    # Clean column names
    raw_eeg.columns = raw_eeg.columns.str.strip()
    print(raw_eeg.columns.tolist())

    # Convert `time` column to timedelta
    raw_eeg['time'] = pd.to_timedelta(raw_eeg['time'], unit='s')

    # Set time as index
    raw_eeg.set_index('time', inplace=True)

    # Drop unwanted columns
    raw_eeg.drop(columns=['obs', 'Derived', 'totPwr', 'class'], inplace=True, errors='ignore')

    # Sort by time
    raw_eeg.sort_index(inplace=True)

    # Resample to 32 Hz (every 31.25 ms), interpolate missing values
    raw_eeg = raw_eeg.resample('31.25ms').mean().interpolate()

    # Normalize with Min-Max Scaling
    scaler = MinMaxScaler()
    scaled_eeg = pd.DataFrame(
        scaler.fit_transform(raw_eeg),
        columns=raw_eeg.columns,
        index=raw_eeg.index
    )

    print("Preprocessing EEG data done!")
    print(scaled_eeg.shape, scaled_eeg.columns.tolist())

    # Save processed data
    processed_path = os.environ["PROCESSED_PATH"]
    os.makedirs(processed_path, exist_ok=True)
    output_path = os.path.join(processed_path, "processed_eeg.csv")
    scaled_eeg.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")

    return None




# This we can delete - we are no longer doing a deep learning module.
def preprocess_dl(raw_data):
    """
    This funcitons takes the raw data for deep learning and applies the following preprocessing steps:
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
