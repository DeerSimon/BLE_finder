import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.base import BaseEstimator, ClassifierMixin
import os

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df[["Frame_length", "RSSI", "MAC Period", "Occurrences", "Fingerprint", "Classification"]]
    df["Fingerprint"] = LabelEncoder().fit_transform(df["Fingerprint"])
    df = df.dropna()
    return df

def split_and_scale_data(df):
    X = df.drop(columns=["Classification"])
    y = df["Classification"]
    X_scaled = MinMaxScaler().fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def train_knn(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def train_adaboost(X_train, y_train, n_estimators=100):
    adaboost = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    adaboost.fit(X_train, y_train)
    return adaboost

def train_nearest_centroid(X_train, y_train):
    nc = NearestCentroid()
    nc.fit(X_train, y_train)
    return nc

def train_nearest_neighbors(X_train, n_neighbors):
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_train)
    return nn


class CustomKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, weights):
        self.weights = weights

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _distance(self, a, b):
        return (
            self.weights[0] * abs(a[0] - b[0]) +
            self.weights[1] * abs(a[1] - b[1]) +
            self.weights[2] * abs(a[2] - b[2]) +
            self.weights[3] * abs(a[3] - b[3]) +
            self.weights[4] * (0 if a[4] == b[4] else 1) #+
            #self.weights[5] * abs(a[4] - b[4])
        )

    def predict(self, X):
        predictions = []
        for x in X:
            dists = [self._distance(x, train_x) for train_x in self.X_train]
            nearest_idx = np.argmin(dists)
            predictions.append(self.y_train[nearest_idx])
        return np.array(predictions)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Model Accuracy: {accuracy:.2%}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues' if model_name == 'k-NN' else 'Oranges')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #plt.title(f'{model_name} Confusion Matrix')
    plt.show()
    
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    print(f"Model Accuracy: {accuracy:.2%}")
def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory,'train_dataset', 'grouped_data_truth.csv')
    
    df = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_and_scale_data(df)
    
    knn_model = train_knn(X_train, y_train,2)
    evaluate_model(knn_model, X_test, y_test, "K-NN")
    
    adaboost_model = train_adaboost(X_train, y_train)
    evaluate_model(adaboost_model, X_test, y_test, "AdaBoost")
    
    nc_model = train_nearest_centroid(X_train, y_train)
    evaluate_model(nc_model, X_test, y_test, "Nearest Centroid")
    
    #nn_model = train_nearest_neighbors(X_train)
    #print("NearestNeighbors model trained, but it is unsupervised and does not produce direct classification results.")

if __name__ == "__main__":
    main()
