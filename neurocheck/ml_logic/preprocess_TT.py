import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_eeg_data(csv_path): # This takes raw CSV file from one of the sessions either morning or night and scales - preparing the data for XGBoost.

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
