import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F


class LSTM_CNNClassifier(pl.LightningModule):

    def __init__(self):
        super(LSTM_CNNClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=4,
            batch_first=True,
            # bidirectional=True,
        )

        # Fully connected layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 3)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.50)

    def forward(self, x):
        dx = torch.diff(x, dim=-1, prepend=x[..., :1])  # doing derivative
        # Concatenate the raw ECG signal and its first derivative along the channel axis
        x = torch.stack((x, dx), dim=1)  # Shape (batch_size, 2, seq_length)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))  # Apply the second convolutional layer
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))  # Apply the third convolutional layer
        x = F.max_pool1d(x, 2)

        # Prepare data for LSTM
        x = x.permute(
            0, 2, 1
        )  # Change shape to (batch_size, sequence_length, features)

        x, (hn, cn) = self.lstm(x)  # LSTM output and hidden states
        x = x[:, -1, :]  # Take the last output of the sequence

        x = F.relu(self.fc1(x))  # Fully connected layers for classification
        x = self.fc2(x)
        return x


class EcgDecisionClass:
    """
    x_test is the testing data, where each row is the ECG beats segmented. \n
    model is the saved classification model. \n

    It will return "predicted_labels" which is the pac/pvc vs normal decision

    Classify ECG beats as normal or abnormal (PAC/PVC) using a pretrained classification model.

    Parameters:
    ----------
    x_test : np.ndarray
        The testing data array where each row corresponds to an ECG beat segment.
    model : Any
        The pretrained PyTorch model used for classification.

    Returns:
    -------
    np.ndarray
        An array of predicted labels, where each label represents the decision for the corresponding ECG beat:
        '0' for normal, '1' for PAC/PVC (abnormal) depending on model configuration.

    Notes:
    ------
    - The function converts the input data to a PyTorch tensor and prepares it for model inference.
    - A DataLoader is used to handle batches of data for potentially large datasets.
    - Predictions are obtained from the model output, which are typically logits that are processed through a softmax
      function to derive probabilities.
    - The function assumes that the model outputs logits directly related to the class probabilities of normal, PAC,
      and PVC beats.
    """

    def __init__(self, model_path):
        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()

    def ecg_abnormal_decision(self, x_test):
        """
        x_test is the testing data, where each row is the ECG beats segmented. \n

        It will return "predicted_labels" which is the pac/pvc vs normal decision

        Classify ECG beats as normal or abnormal (PAC/PVC) using a pretrained classification model.

        Parameters:
        ----------
        x_test : np.ndarray
            The testing data array where each row corresponds to an ECG beat segment.

        Returns:
        -------
        np.ndarray
            An array of predicted labels, where each label represents the decision for the corresponding ECG beat:
            '0' for normal, '1' for PAC/PVC (abnormal) depending on model configuration.

        Notes:
        ------
        - The function converts the input data to a PyTorch tensor and prepares it for model inference.
        - A DataLoader is used to handle batches of data for potentially large datasets.
        - Predictions are obtained from the model output, which are typically logits that are processed through a softmax
        function to derive probabilities.
        - The function assumes that the model outputs logits directly related to the class probabilities of normal, PAC,
        and PVC beats.
        """
        threshold = 0.70

        if x_test is None or len(x_test) == 0:
            return [], []

        x_test = np.array(x_test)

        # doing normalization
        min_vals = np.min(x_test, axis=1, keepdims=True)
        max_vals = np.max(x_test, axis=1, keepdims=True)
        x_test = (x_test - min_vals) / (max_vals - min_vals + 1e-8)

        test_data_tensor = torch.tensor(x_test, dtype=torch.float32)
        outputs = self.model(test_data_tensor)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get the maximum probability and corresponding class
        max_prob, predicted = torch.max(probabilities, dim=1)

        # Assign to 4th class ("none of the above") if max probability is below the threshold
        predicted = torch.where(max_prob < threshold, torch.tensor(3), predicted)

        predicted_labels = np.array(predicted)

        return predicted_labels, max_prob
