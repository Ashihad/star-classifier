import torch
import torch.nn as nn
import torch.optim as optim

from src.stellar_classifier_base import StellarClassifierBase
from src.utilities import get_common_paths

paths = get_common_paths()

class NNModel(nn.Module):
  def __init__(self, input_size, num_classes):
    super(NNModel, self).__init__()

    dropout = 0.67
    hidden_size1 = 256
    hidden_size2 = 256
    hidden_size3 = 128

    self.dropout = nn.Dropout(dropout)
    self.layers = nn.Sequential(
        nn.Linear(input_size, hidden_size1),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(hidden_size1),
        self.dropout,
        nn.Linear(hidden_size1, hidden_size2),
        nn.LeakyReLU(0.01),
        nn.BatchNorm1d(hidden_size2),
        self.dropout,
        nn.Linear(hidden_size2, hidden_size3),
        nn.LeakyReLU(0.02),
        nn.BatchNorm1d(hidden_size3),
        self.dropout,
        nn.Linear(hidden_size3, num_classes),
    )
        
  def forward(self, x):
    return self.layers(x)

class StellarClassifierNN(StellarClassifierBase):
  def __init__(self):
    """Modify this to change model hyperparameters"""
    super(StellarClassifierNN, self).__init__()
    self.batch_size = 32
    self.learning_rate = 0.002
    self.num_epochs = 200
    self.smote = True
    self.feature_columns = ['u', 'g', 'r', 'i', 'z', 'redshift']
    self.output_dir = paths['nn_dir_path']

    self.logger.info("StellarClassifierNN initialized")

  def set_params(self):
    """Modify this to change model itself, criterion function and optimizer"""
    num_classes = len(self.label_encoder.classes_)
    input_size = len(self.feature_columns)

    self.model = NNModel(input_size, num_classes).to(self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

def main():
  classifier = StellarClassifierNN()
  classifier.run()

if __name__ == "__main__":
  main()
