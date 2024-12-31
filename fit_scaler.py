# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Load the combined dataset
# combined_data_file = 'combined_voice_features.csv'
# combined_data = pd.read_csv(combined_data_file)

# # Separate features (X) and labels (y)
# X = combined_data.drop(columns=['label'])  # All columns except 'label'

# # Fit the scaler
# scaler = StandardScaler()
# scaler.fit(X)  # Fit the scaler on the features

# # Save the fitted scaler
# import joblib
# scaler_file = 'scaler.pkl'
# joblib.dump(scaler, scaler_file)
# print(f"Scaler saved to {scaler_file}")


# import joblib

# # Load the fitted scaler
# scaler_file = 'scaler.pkl'
# scaler = joblib.load(scaler_file)

# # Use the loaded scaler in your test_voice function
# test_voice(model, scaler=scaler)

