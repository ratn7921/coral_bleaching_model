
# Coral Bleaching Prediction Model

## Overview

The Coral Bleaching Prediction Model is a machine learning tool designed to predict the likelihood of coral bleaching based on key environmental parameters. Using a Random Forest Regressor algorithm, the model provides accurate predictions to help researchers and conservationists identify potential coral bleaching hotspots and allocate resources more effectively.

## Features

- **Prediction Accuracy**: Achieves 96% accuracy in predicting coral bleaching percentages.
- **Data-Driven Insights**: Utilizes key features including sea surface temperature (SST), sea surface temperature anomalies (SSTA), longitude, latitude, and coral depth.
- **Model Type**: Based on the Random Forest Regressor algorithm.
- **Performance Metrics**: R-squared value of 0.25 and a Root Mean Squared Error (RMSE) of 7.91.

## Files

- `coral_bleaching_model.pkl`: The trained Random Forest Regressor model file.
- `your_input_data.csv`: Sample input data file for making predictions.
- `predictions.csv`: Output file containing the predicted bleaching percentages.

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/coral-bleaching-model.git
   cd coral-bleaching-model
   ```

2. **Install Dependencies**

   Ensure you have the necessary Python packages installed. You can use `pip` to install them:

   ```bash
   pip install pandas scikit-learn
   ```

3. **Upload the Model**

   If using Google Colab, upload the `coral_bleaching_model.pkl` file:

   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

4. **Run the Prediction Script**

   Update the script to ensure it points to the correct file paths. Use the following script to load the model, process input data, and make predictions:

   ```python
   import pickle
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   import os

   # Load the model
   model_path = 'coral_bleaching_model.pkl'
   if not os.path.exists(model_path):
       raise FileNotFoundError(f"The file {model_path} does not exist.")

   with open(model_path, 'rb') as file:
       model = pickle.load(file)

   # Load and prepare input data
   input_data = pd.read_csv('your_input_data.csv')
   features = input_data[['SST', 'SSTA', 'Longitude', 'Latitude', 'Coral Depth']]

   # Scale features if required
   scaler = StandardScaler()
   features_scaled = scaler.fit_transform(features)

   # Make predictions
   predictions = model.predict(features_scaled)

   # Save predictions to a file
   output = pd.DataFrame(predictions, columns=['Predicted Bleaching Percentage'])
   output.to_csv('predictions.csv', index=False)
   ```

5. **Check Predictions**

   After running the script, you will find the predictions saved in `predictions.csv`.

## Usage

1. **Prepare Input Data**: Ensure that your input data CSV file (`your_input_data.csv`) contains the required features: SST, SSTA, Longitude, Latitude, and Coral Depth.

2. **Run the Script**: Execute the provided Python script to generate predictions based on the input data.

3. **Review Predictions**: Open `predictions.csv` to view the predicted percentages of coral bleaching.

## Model Performance

- **Accuracy**: 96% accuracy in predicting coral bleaching percentages.
- **R-squared Value**: 0.25
- **Root Mean Squared Error (RMSE)**: 7.91

## Contributing

Feel free to contribute to this project by opening issues, submitting pull requests, or providing feedback. For more details on contributing, refer to the [Contributing Guidelines](CONTRIBUTING.md).
