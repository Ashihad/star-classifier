import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, validation_curve

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.stellar_classifier_base import StellarClassifierBase
from src.utilities import get_common_paths

paths = get_common_paths()

class StellarClassifierKNN(StellarClassifierBase):
  """This version heavily modifies StellarClassifierBase since it uses sklearn instead of pytorch"""
  def __init__(self):
    """Modify this to change model hyperparameters"""
    super(StellarClassifierKNN, self).__init__()
    self.smote = True
    self.feature_columns = ['u', 'g', 'r', 'i', 'z', 'redshift']
    self.output_dir = paths['knn_dir_path']

    self.logger.info("StellarClassifierKNN initialized")

  # model does not use pytorch interface
  def set_params(self):
    pass

  def find_optimal_k(self, X_train, y_train, k_range):
    self.logger.info("Finding optimal k value...")
    
    k_scores = []
    for k in tqdm(k_range):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())

    # Find best k
    optimal_k = k_range[np.argmax(k_scores)]
    self.logger.info(f"Optimal k value: {optimal_k}")

    return optimal_k

  def plot_training_vs_validation(self, k_range, optimal_k, train_scores_mean, train_scores_std, val_scores_mean, val_scores_std):
    self.logger.info("Plotting training vs validation")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(k_range, train_scores_mean, 'o-', color='blue', label='Training Accuracy')
    ax.fill_between(k_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    ax.plot(k_range, val_scores_mean, 'o-', color='red', label='Validation Accuracy')
    ax.fill_between(k_range, val_scores_mean - val_scores_std,
                    val_scores_mean + val_scores_std, alpha=0.1, color='red')
    ax.axvline(x=optimal_k, color='green', linestyle='--', label=f'Optimal k={optimal_k}')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Accuracy')
    ax.set_title('KNN: Training vs Validation Accuracy')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(self.output_dir + "/training_plots.png")
    plt.show()

    self.logger.info("Plotting done")

  def run(self):
    df = self.load_data()
    X_train, X_test, y_train, y_test = self.prepare_data(df)

    k_range = range(1, min(21, len(X_train)))

    optimal_k = self.find_optimal_k(X_train, y_train, k_range)

    self.logger.info("Generating validation curve...")
    train_scores, val_scores = validation_curve(
      KNeighborsClassifier(), X_train, y_train, 
      param_name='n_neighbors', param_range=k_range, 
      cv=5, scoring='accuracy', n_jobs=-1
    )
    self.logger.info("Validation curve generation done")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    knn_model = KNeighborsClassifier(n_neighbors=optimal_k)

    self.logger.info("Training...")
    knn_model.fit(X_train, y_train)
    self.logger.info("Training done")

    self.logger.info("Predicting...")
    y_pred = knn_model.predict(X_test)
    self.logger.info("Predicting done")

    test_accuracy = accuracy_score(y_test, y_pred)

    self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    self.logger.info("Classification Report:")
    self.logger.info(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

    self.plot_training_vs_validation(k_range, optimal_k, train_scores_mean, train_scores_std, val_scores_mean, val_scores_std)
    self.plot_confusion_matrix(y_test, y_pred)

def main():
  classifier = StellarClassifierKNN()
  classifier.run()

if __name__ == "__main__":
  main()
