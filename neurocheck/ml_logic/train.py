import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_eeg_data, filter_to_eeg_channels
from neurocheck.ml_logic.model import initialize_xgb_model, train_model, evaluate_model
from registry import save_results, save_model

# Load raw data
df = pd.read_csv('/Users/majamielke/code/Cdennis51/Neurocheck/raw_data/MEFAR_preprocessed/MEFAR_MID.csv')

# Set target:
y = df['class']

# Set X and remove other features present in the MEFAR MID dataset.
X = filter_to_eeg_channels(df)

# Train_Test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize the model
model = initialize_xgb_model()

# Train and save
train_model(X_train,y_train,model)

# Save the results:
#save_results()

# Save the model to MLFlow:
save_model(model)
