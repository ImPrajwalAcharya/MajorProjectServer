from functools import lru_cache
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn



# Define your AirModel class
class AirModel(nn.Module):
    def __init__(self, num_features):
        super(AirModel, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=200, num_layers=3, batch_first=True)
        self.linear = nn.Linear(200, 5)  # Output size is now 5 for the five features

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
# Load the test data (replace 'new_data.csv' with the path to your new data file)

def load_csv():
    df = pd.read_csv('Weatherhourly.csv')
    timeseries = df[["PRECTOTCORR", "QV2M", "T2M", "PS", "WS10M"]].values.astype('float32')
    return timeseries
# new_timeseries = normalize(new_timeseries)

# Create the dataset for prediction (similar to the create_dataset function used for training)
def create_prediction_dataset(): 
    timeseries = load_csv()
    # lookback = 10
    # X_pred = []
    # for i in range(len(dataset) - lookback):
    #     feature = dataset[i:i + lookback]
    #     X_pred.append(feature)
    
    sequence_length = 10

# Prepare the input data for future prediction
    input_data = []
    for i in range(len(timeseries) - sequence_length):
        input_data.append(timeseries[i:i+sequence_length])

    input_data = np.array(input_data)

# Reshape the input data to (number of sequences, sequence length, number of features)
    input_data = input_data.reshape(-1, sequence_length, 5)

# Convert the NumPy array to a PyTorch tensor
    input_data_tensor = torch.tensor(input_data)
    return input_data_tensor


# @contextmanager
# def make_prediction(model: AirModel, dataset):
#     """
#     dataset is of tensor type
#     """
    
#     with torch.no_grad():
#         predictions = model(dataset)
#         yield predictions[:, -1, :]


@lru_cache(maxsize=1)
def load_model() -> AirModel:
    checkpoint_path = "./trainedmodelfor5.pth"
    model = AirModel(num_features=5)
    model.load_state_dict(torch.load(checkpoint_path))
    return model













# Assuming 'Weatherhourly.csv' contains columns 'PRECTOTCORR', 'QV2M', 'T2M', 'PS', and 'WS10M'


# Define the sequence length (number of time steps to use for prediction)


# Load the trained model from the checkpoint file


# Predict the next 10 steps for all five parameters
@contextmanager
def make_prediction(model: AirModel, input_data_tensor):
    with torch.no_grad():
        model.eval()
        future_predictions = []

    # Use the last sequence from the input data as the initial input for prediction
        last_sequence = input_data_tensor[-1:]

        for _ in range(10):
        # Predict the next time step
            next_prediction = model(last_sequence)

        # Append the prediction to the results
            future_predictions.append(next_prediction[:, -1])

        # Update the last_sequence with the latest prediction for autoregressive prediction
            last_sequence = torch.cat((last_sequence[:, 1:], next_prediction[:, -1:]), dim=1)

    future_predictions = torch.cat(future_predictions, dim=1).numpy()
    yield future_predictions
