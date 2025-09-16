import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta


def predict_for_date(input_date, csv_file='gold_prices_multi_year.csv',
                     model_file='gold_price_lstm_multi_output_model.h5'):
    # Load and preprocess data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Convert input date to datetime
    target_date = pd.to_datetime(input_date)

    # Check if we have enough data (at least 60 days before the target date)
    if target_date not in df.index:
        print(f"Error: Date {target_date.strftime('%Y-%m-%d')} not found in the dataset.")
        return

    # Find the index of the target date
    target_idx = df.index.get_loc(target_date)

    if target_idx < 60:
        print(f"Error: Not enough historical data before {target_date.strftime('%Y-%m-%d')}.")
        print(f"Need 60 days of data, but only {target_idx} days available.")
        return

    # Define features
    FEATURES = ['High', 'Low', 'Open', 'Close']
    OUTPUT_FEATURES = ['High', 'Low', 'Open', 'Close']
    output_idx = [FEATURES.index(c) for c in OUTPUT_FEATURES]

    # Load the trained model
    model = load_model(model_file)

    # Prepare scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[FEATURES].values)

    # Function to inverse transform predictions
    def inverse_transform_multi_predictions(prediction, scaler, data, output_idx):
        n_features = data.shape[1]
        dummy_array = np.zeros((1, n_features))

        for i, idx in enumerate(output_idx):
            dummy_array[0, idx] = prediction[i]

        inverse_scaled = scaler.inverse_transform(dummy_array)
        return inverse_scaled[0, output_idx]

    # Get the 60 days before the target date
    previous_60_days = scaled_data[target_idx - 60:target_idx]

    # Prepare input for prediction
    x_input = previous_60_days.reshape(1, 60, len(FEATURES))

    # Predict the target date
    prediction = model.predict(x_input, verbose=0)[0]

    # Convert prediction to actual values
    predicted_prices = inverse_transform_multi_predictions(
        prediction, scaler, df[FEATURES].values, output_idx
    )

    # Create results DataFrame
    predicted_df = pd.DataFrame([predicted_prices], columns=OUTPUT_FEATURES, index=[target_date])

    # Get actual values for comparison
    actual_values = df.loc[target_date, OUTPUT_FEATURES]

    # Print the prediction
    print("Gold Price Prediction:")
    print("=" * 60)
    print(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Predicted High: ${predicted_df['High'].iloc[0]:.2f}")
    print(f"Predicted Low: ${predicted_df['Low'].iloc[0]:.2f}")
    print(f"Predicted Open: ${predicted_df['Open'].iloc[0]:.2f}")
    print(f"Predicted Close: ${predicted_df['Close'].iloc[0]:.2f}")
    print()
    print("Actual Values (for comparison):")
    print(f"Actual High: ${actual_values['High']:.2f}")
    print(f"Actual Low: ${actual_values['Low']:.2f}")
    print(f"Actual Open: ${actual_values['Open']:.2f}")
    print(f"Actual Close: ${actual_values['Close']:.2f}")

    # Calculate errors
    errors = {}
    for feature in OUTPUT_FEATURES:
        errors[feature] = abs(predicted_df[feature].iloc[0] - actual_values[feature])

    print()
    print("Prediction Errors:")
    for feature, error in errors.items():
        print(f"{feature} Error: ${error:.2f}")

    # Plot the last 60 days and the prediction
    plt.figure(figsize=(14, 10))

    # Plot historical data
    historical_dates = df.index[target_idx - 60:target_idx]

    # Create subplots for each feature
    for i, feature in enumerate(OUTPUT_FEATURES):
        plt.subplot(2, 2, i + 1)
        plt.plot(historical_dates, df[feature][target_idx - 60:target_idx],
                 label=f'Historical {feature}', color='blue', linewidth=2)

        # Plot prediction
        plt.plot(predicted_df.index, predicted_df[feature], 'ro',
                 label=f'Predicted {feature}', markersize=8)

        # Plot actual value
        plt.plot([target_date], [actual_values[feature]], 'go',
                 label=f'Actual {feature}', markersize=8)

        plt.title(f'{feature} Price - Prediction vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return predicted_df, actual_values, errors


# Example usage
if __name__ == "__main__":
    # Get input date from user
    input_date = input("Enter the date for prediction (format: MM/DD/YYYY): ")

    # Predict for the input date
    try:
        predicted, actual, errors = predict_for_date(input_date)
    except Exception as e:
        print(f"An error occurred: {e}")