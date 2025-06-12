import torch
import torch.nn as nn
import torch.optim as optim

from src.stellar_classifier_base import StellarClassifierBase
from src.utilities import get_common_paths

paths = get_common_paths()

class LogisticRegressionModel(nn.Module):
  def __init__(self, input_size, num_classes):
    super(LogisticRegressionModel, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)
        
  def forward(self, x):
    x = self.linear(x)
    x = torch.sigmoid(x)
    return x

class StellarClassifierLogReg(StellarClassifierBase):
  def __init__(self):
    """Modify this to change model hyperparameters"""
    super(StellarClassifierLogReg, self).__init__()
    self.batch_size = 32
    self.learning_rate = 0.01
    self.num_epochs = 200
    self.smote = True
    self.feature_columns = ['u', 'g', 'r', 'i', 'z', 'redshift']
    self.output_dir = paths['logreg_dir_path']

    self.logger.info("StellarClassifierLogReg initialized")

  def set_params(self):
    """Modify this to change model itself, criterion function and optimizer"""
    num_classes = len(self.label_encoder.classes_)
    input_size = len(self.feature_columns)

    self.model = LogisticRegressionModel(input_size, num_classes).to(self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

def main():
  classifier = StellarClassifierLogReg()
  classifier.run()

if __name__ == "__main__":
  main()
