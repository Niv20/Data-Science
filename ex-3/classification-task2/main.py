# DELETE ME
"""
cd /Users/niv/Documents/GitHub/Data-Science/ex-3/classification-task2
python main.py
🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮🐮
"""

import time
import json
import pandas as pd
from pandas import DataFrame

from sklearn.metrics import accuracy_score
from typing import List

# We add this:
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


# Helper function 
def get_radius_predictions(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, radius: float) -> List:
    """
    In this function, we will implement Radius Neighbors Classification (NNR).
    Unlike KNN, which finds fixed k neighbors, this algorithm finds ALL neighbors 
    within a fixed distance (radius).
    
    Args:
        X_train: Training features (scaled).
        y_train: Training labels.
        X_test: Test features (scaled).
        radius: The distance threshold to consider a point as a neighbor.
        
    Returns:
        List of predicted labels.
    """
    predictions = []
    
    # Pre-calculate the global majority class from the training set.
    # This is used as a fallback strategy when a test instance has NO neighbors within the radius.
    # (i.e., it is an outlier in the feature space).
    global_most_common = Counter(y_train).most_common(1)[0][0]
    
    for i in range(len(X_test)):
        test_instance = X_test[i]
        
        # --- 1. Calculate Euclidean Distances ---
        # Calculate distance from this test instance to ALL training instances.
        # Formula: sqrt(sum((x_train - x_test)^2))
        distances = np.sqrt(((X_train - test_instance) ** 2).sum(axis=1))
        
        # --- 2. Filter Neighbors by Radius ---
        # Crucial difference from KNN: We do NOT sort and pick k!
        # We simply select all indices where (distance <= radius).
        within_radius_indices = np.where(distances <= radius)[0]
        
        # --- 3. Voting / Fallback ---
        if len(within_radius_indices) > 0:
            # If neighbors exist, take their labels
            neighbor_labels = y_train[within_radius_indices]
            # Perform Majority Vote among the neighbors in the radius
            vote_result = Counter(neighbor_labels).most_common(1)[0][0]
            predictions.append(vote_result)
        else:
            # Logic for "Zero Neighbors":
            # If the radius is too small or the point is an outlier, no neighbors are found.
            # We assign the Global Majority Class (safest baseline).
            predictions.append(global_most_common)
            
    return predictions

def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    """
    Main pipeline for Radius Neighbors Classification.
    1. Loads and Scales Data.
    2. Determines candidate radii based on data distribution (percentiles).
    3. Tunes the 'radius' hyperparameter on the Validation set.
    4. Retrains on Train+Val and predicts on Test.
    """

    #  the data_tst dataframe should only(!) be used for the final predictions your return
    print(f'starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

# ---------------------------------------------------------
    # Step 1: Data Loading
    # ---------------------------------------------------------
    df_train = pd.read_csv(data_trn)
    df_val = pd.read_csv(data_vld)

    # Convert to numpy arrays for efficiency in our manual function
    X_train_raw = df_train.drop(['class'], axis=1).values
    y_train = df_train['class'].values
    
    X_val_raw = df_val.drop(['class'], axis=1).values
    y_val = df_val['class'].values
    
    X_test_raw = df_tst.values
    
    # ---------------------------------------------------------
    # Step 2: Feature Scaling
    # ---------------------------------------------------------
    # Critical for distance-based methods.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # ---------------------------------------------------------
    # Step 3: Define Hyperparameter Search Space (Candidate Radii)
    # ---------------------------------------------------------
    # Unlike 'k' (where 1, 3, 5 are obvious choices), 'radius' depends on the data scale.
    # To find good candidates, we sample distances from the training set itself.
    
    # Take a random sample of 500 training points to estimate density
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(X_train_scaled), min(500, len(X_train_scaled)), replace=False)
    X_sample = X_train_scaled[sample_indices]
    
    # Calculate pairwise distances within this sample to understand the "typical distance"
    # shape: (N_sample, N_sample)
    dists_sample = np.sqrt(((X_sample[:, np.newaxis, :] - X_sample[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Flatten and ignore self-distances (0.0)
    dists_flat = dists_sample[dists_sample > 0]
    
    # Define candidates as percentiles of the distance distribution.
    # e.g., 5th percentile = a tight radius (very local).
    # 30th percentile = a broad radius.
    percentiles = [2, 5, 10, 15, 20, 30] 
    candidate_radii = np.unique(np.percentile(dists_flat, percentiles))
    
    print(f"Candidate radii (based on train set density): {np.round(candidate_radii, 4)}")

    # ---------------------------------------------------------
    # Step 4: Hyperparameter Tuning (Find Best Radius)
    # ---------------------------------------------------------
    best_radius = candidate_radii[0]
    best_acc = -1
    
    for r in candidate_radii:
        # Predict on Validation set
        val_preds = get_radius_predictions(X_train_scaled, y_train, X_val_scaled, r)
        
        # Calculate Accuracy
        acc = accuracy_score(y_val, val_preds)
        
        print(f"Radius: {r:.4f} -> Val Acc: {acc:.4f}") # Dear Bodek, you can uncomment this line out if you want 👈🏼
        
        if acc > best_acc:
            best_acc = acc
            best_radius = r
            
    print(f"Best radius found: {best_radius:.4f} with Validation Accuracy: {best_acc:.4f}")

    # ---------------------------------------------------------
    # Step 5: Final Prediction
    # ---------------------------------------------------------
    # Combine Train + Validation for the final model (More data = Better density estimation)
    X_full_scaled = np.vstack((X_train_scaled, X_val_scaled))
    y_full = np.concatenate((y_train, y_val))
    
    # Predict on Test using the optimal radius
    final_predictions = get_radius_predictions(X_full_scaled, y_full, X_test_scaled, best_radius)
    
    return final_predictions

# Our student ids 🪪
students = {'id1': '212136287', 'id2': '207298514'}

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
