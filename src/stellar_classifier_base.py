import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import logging
from abc import ABC, abstractmethod

from .utilities import get_common_paths

class StellarClassifierBase(ABC):
  def __init__(self):
    """After this init call, you need to define self.batch_size, self.learning_rate and self.num_epochs manually"""
    self.logger = logging.getLogger("main")
    self.model = None
    self.scaler = StandardScaler()
    self.label_encoder = LabelEncoder()
    self.feature_columns = ['u', 'g', 'r', 'i', 'z']
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.logger.info(f"Using device: {self.device}")

    paths = get_common_paths()
    self.data_path = paths['data_path']
    self.output_dir = paths['default_dir_path']

    self.logger.info("StellarClassifierBase initialized")

  @abstractmethod
  def set_params():
    """Override this method to initialize self.model, self.criterion and self.optimizer"""
    pass

  def load_data(self):
    self.logger.info("Loading data...")

    df = pd.read_csv(self.data_path)

    # remove invalid rows
    df = df[df["u"] != -9999]

    # take only light parameters and labels
    df = df[self.feature_columns + ["class"]]
    self.logger.info("Data loading done")
    return df

  def prepare_data(self, df):
    """Prepare and preprocess the data"""
    self.logger.info("Preparing data...")

    X = df[self.feature_columns].values
    y = df['class'].values
    
    # encode labels
    y_encoded = self.label_encoder.fit_transform(y)
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded
    )
    # scale features
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)

    self.logger.info("Data preparation done")
    
    return X_train_scaled, X_test_scaled, y_train, y_test
    
  def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=32):
    """Create PyTorch data loaders"""
    # convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(self.device)
    X_test_tensor = torch.FloatTensor(X_test).to(self.device)
    y_train_tensor = torch.LongTensor(y_train).to(self.device)
    y_test_tensor = torch.LongTensor(y_test).to(self.device)
    
    # create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
    
  def train_model(self, train_loader):
    """Train model"""
    self.logger.info("Training model...")

    train_losses = []
    train_accuracies = []
    
    for epoch in range(self.num_epochs):
      self.model.train()
      epoch_loss = 0.0
      correct = 0
      total = 0
      
      for batch_features, batch_labels in train_loader:
        self.optimizer.zero_grad()
        
        outputs = self.model(batch_features)
        loss = self.criterion(outputs, batch_labels)
        
        loss.backward()
        self.optimizer.step()
        
        epoch_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
      
      avg_loss = epoch_loss / len(train_loader)
      accuracy = 100 * correct / total
      
      train_losses.append(avg_loss)
      train_accuracies.append(accuracy)
      
      if (epoch + 1) % 20 == 0:
        self.logger.info(f'Epoch [{epoch+1:03}/{self.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    self.logger.info("Training finished")
    
    return train_losses, train_accuracies
    
  def evaluate_model(self, test_loader):
    """Evaluate model"""
    self.logger.info("Evaluating model...")

    self.model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
      for batch_features, batch_labels in test_loader:
        outputs = self.model(batch_features)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    
    self.logger.info("Model evaluation done")

    return np.array(all_predictions), np.array(all_labels)
    
  def predict(self, X):
    """Make predictions on new data"""
    self.model.eval()
    X_scaled = self.scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(self.device)
    
    with torch.no_grad():
      outputs = self.model(X_tensor)
      probabilities = torch.softmax(outputs, dim=1)
      _, predicted = torch.max(outputs, 1)
    
    # convert back to original labels
    predicted_labels = self.label_encoder.inverse_transform(predicted.cpu().numpy())
    probabilities_np = probabilities.cpu().numpy()
    
    return predicted_labels, probabilities_np
    
  def plot_training_history(self, losses, accuracies):
    """Plot training loss and accuracy"""
    self.logger.info("Plotting training loss and accuracy...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(self.output_dir + "/training_plots.png")
    plt.show()

    self.logger.info("Plotting done")
    
  def plot_confusion_matrix(self, y_true, y_pred):
    """Plot confusion matrix"""
    self.logger.info("Plotting confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(self.output_dir + "/confusion_matrix.png")
    plt.show()

    self.logger.info("Plotting done")

  def run(self):
    df = self.load_data()
    X_train, X_test, y_train, y_test = self.prepare_data(df)
    train_loader, test_loader = self.create_data_loaders(
      X_train, X_test, y_train, y_test, batch_size=self.batch_size
    )

    self.set_params()
    losses, accuracies = self.train_model(train_loader)

    y_pred, y_true = self.evaluate_model(test_loader)
    test_accuracy = accuracy_score(y_true, y_pred)

    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_))

    self.plot_training_history(losses, accuracies)
    self.plot_confusion_matrix(y_true, y_pred)