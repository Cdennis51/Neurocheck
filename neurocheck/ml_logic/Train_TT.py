import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess_TT import preprocess_eeg_data, filter_to_eeg_channels
from model_TT import initialize_xgb_model, train_model, evaluate_model
from registry import save_results, save_model

# Load raw data
df = pd.read_csv('../raw_data/MEFAR_preprocessed/MEFAR_MID.csv')

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
save_results()

# Save the model to MLFlow:
save_model(model)
