import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def filter_to_eeg_channels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops non-EEG physiological signals from the MEFAR MID dataset.

    Keeps only EEG channels (removes BVP, EDA, TEMP, Acc, HR).

    Parameters:
    - df: DataFrame containing all physiological signals

    Returns:
    - EEG-only DataFrame
    """
    columns_to_drop = ['BVP', 'EDA', 'TEMP', 'AccX', 'AccY', 'AccZ', 'HR', 'class']
    filtered_df = df.drop(columns=columns_to_drop, errors='ignore')
    filtered_df.columns = filtered_df.columns.str.strip()
    return filtered_df

def preprocess_eeg_csv(csv_path: str) -> pd.DataFrame:
    """
    Takes a file path to a raw EEG CSV and returns preprocessed DataFrame.
    """
    raw_eeg = pd.read_csv(csv_path)
    return preprocess_eeg_df(raw_eeg)

def preprocess_eeg_df(raw_eeg: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw EEG DataFrame and returns a cleaned, resampled, and scaled version.
    """
    raw_eeg.columns = raw_eeg.columns.str.strip()
    raw_eeg['time'] = pd.to_timedelta(raw_eeg['time'], unit='s')
    raw_eeg.set_index('time', inplace=True)
    raw_eeg.drop(columns=['obs', 'Derived', 'totPwr', 'class'], inplace=True, errors='ignore')
    raw_eeg.sort_index(inplace=True)
    raw_eeg = raw_eeg.resample('31.25ms').mean().interpolate()

    scaler = MinMaxScaler()
    scaled_eeg = pd.DataFrame(
        scaler.fit_transform(raw_eeg),
        columns=raw_eeg.columns,
        index=raw_eeg.index
    )

    return scaled_eeg

# def preprocess_eeg_data(csv_path): # This takes raw CSV file from one of the sessions either morning or night and scales - preparing the data for XGBoost.


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

    return scaled_eeg
