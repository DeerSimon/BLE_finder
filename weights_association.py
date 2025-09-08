import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution
import os

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Build the file path
csv_path = os.path.join(current_dir, 'train_dataset', 'grouped_data_truth.csv') #trainset
csv_path_test = os.path.join(current_dir, 'train_dataset', 'test_ground_class.csv')


df = pd.read_csv(csv_path)
test_df = pd.read_csv(csv_path_test)
# Drop rows without classification
df = df.dropna(subset=['Classification','Time Window'])
test_df = test_df.dropna(subset=['Classification', 'Time Window'])
# Rename for consistency
df = df.rename(columns={
    'Frame_length': 'frame_length',
    'RSSI': 'signal',
    'MAC Period': 'mac period',
    'Occurrences': 'occurrences',
    'Fingerprint': 'fingerprint',
    #'Time Window': 'last_seen',
    'Classification': 'label'
})

test_df = test_df.rename(columns={
    'Frame_length': 'frame_length',
    'RSSI': 'signal',
    'MAC Period': 'mac period',
    'Occurrences': 'occurrences',
    'Fingerprint': 'fingerprint',
    #'Time Window': 'last_seen',
    'Classification': 'label'
})

# Normalize numeric features
features = ['frame_length', 'signal', 'mac period', 'occurrences']#, 'last_seen']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
test_df[features] = scaler.transform(test_df[features])

# Encode fingerprint
fingerprint_encoder = LabelEncoder()
df['fingerprint_encoded'] = fingerprint_encoder.fit_transform(df['fingerprint'].astype(str))
test_df['fingerprint_encoded'] = fingerprint_encoder.transform(test_df['fingerprint'].astype(str))

# Final feature matrix (last_seen already normalized)
X = df[features + ['fingerprint_encoded']].values
Z_test = test_df[features + ['fingerprint_encoded']].values
y = df['label'].values
z_test = test_df['label'].values

# Custom distance function
def distance(a, b, weights):
    return (
        weights[0] * abs(a[0] - b[0]) +  # frame_length
        weights[1] * abs(a[1] - b[1]) +  # signal
        weights[2] * abs(a[2] - b[2]) +  # mac period
        weights[3] * abs(a[3] - b[3]) +  # occurrences
        weights[4] * (0 if a[4] == b[4] else 1) #+  # fingerprint
        #weights[5] * abs(a[4] - b[4])  # last_seen
    )

# Leave-one-out 1-NN accuracy
def leave_one_out_accuracy(weights, K=2):
    predictions = []
    for i in range(len(X)):
        test_point = X[i]
        rest = np.delete(X, i, axis=0)
        rest_labels = np.delete(y, i)

        # Distances to all others
        dists = np.array([distance(test_point, other, weights) for other in rest])
        nearest_indices = np.argsort(dists)[:K]
        nearest_labels = rest_labels[nearest_indices]

        # Majority vote
        counts = np.bincount(nearest_labels)
        predicted_label = np.argmax(counts)
        predictions.append(predicted_label)

    return accuracy_score(y, predictions)

# Objective to minimize (negative accuracy)
def objective(weights):
    return -leave_one_out_accuracy(weights)

# Bounds for weights [0,10] for each of the 6 features
bounds = [(0, 10)] * 6

# Run optimization
result = differential_evolution(objective, bounds, seed=42, strategy='best1bin', disp=True)

# Final weights and accuracy
best_weights = result.x
best_accuracy = -result.fun


weights ={
    'frame_length': best_weights[0],
    'signal': best_weights[1],
    'mac period': best_weights[2],
    'occurrences': best_weights[3],
    'fingerprint': best_weights[4]#,
    #'last_seen': best_weights[5]
    
}
# Output
print("\nLearned Feature Weights:")
print(f"Frame Length      (w1): {best_weights[0]:.4f}")
print(f"Signal Strength   (w2): {best_weights[1]:.4f}")
print(f"MAC Period        (w3): {best_weights[2]:.4f}")
print(f"Occurrences       (w4): {best_weights[3]:.4f}")
print(f"Fingerprint Match (w5): {best_weights[4]:.4f}")
#print(f"Last Seen         (w6): {best_weights[5]:.4f}")
print(f"\nLeave-One-Out Accuracy: {best_accuracy:.4f}")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from knn import evaluate_model, CustomKNN  # Assuming your function is here
#
## Create model with learned weights
knn_model = CustomKNN(weights=best_weights)
knn_model.fit(X_train, y_train)
#
## Evaluate
evaluate_model(knn_model, X_test, y_test, "Weighted K-NN")
#
Z_train, N_test, z_train, n_test = train_test_split(Z_test, z_test, test_size=0.8, random_state=42)
knn_model = CustomKNN(weights=best_weights)
knn_model.fit(Z_train, z_train)
evaluate_model(knn_model, N_test, n_test, "Weighted K-NN")


weights_path = os.path.join(current_dir, 'weights.csv')
pd.DataFrame(weights, index=[0]).to_csv(weights_path, index=False)