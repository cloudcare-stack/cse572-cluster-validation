import pandas as pd
import numpy as np

'''
Feature Engineering for CGM and Insulin Data

This code processes CGM and insulin data to extract meal-related glucose patterns,
clusters them using KMeans and DBSCAN, and evaluates the clusters using SSE, entropy, and purity.

Purpose: To analyze the relationship between meal carbohydrate intake and subsequent glucose patterns,
and to evaluate the effectiveness of clustering methods in grouping similar glucose responses.

Steps:
1. Load CGM and insulin data from CSV files.
2. Extract meal events based on insulin carb input and create a matrix of glucose values following each meal.
3. Create ground truth bins for carbohydrate intake based on 20-gram intervals.
4. Preprocess the glucose data by imputing missing values and standardizing features.
5. Run KMeans and DBSCAN clustering algorithms on the preprocessed data.
6. Compute evaluation metrics (SSE, entropy, purity) for the clusters.

Technlogies Used:
- Python
- Pandas for data manipulation
- NumPy for numerical operations
- Scikit-learn for clustering and preprocessing is structured into functions for modularity and clarity, 
    with a main function to orchestrate the workflow.
- The results are saved to a CSV file for further analysis.

Author: Colin McAteer

Date: 2024-06-01

'''

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy

# Load data
cgm = pd.read_csv("CGMData.csv")
insulin = pd.read_csv("InsulinData.csv")

# Combine datetime
cgm["datetime"] = pd.to_datetime(cgm["Date"] + " " + cgm["Time"])
insulin["datetime"] = pd.to_datetime(insulin["Date"] + " " + insulin["Time"])

# Sort
cgm = cgm.sort_values("datetime")
insulin = insulin.sort_values("datetime")

# Extract meal events
insulin = insulin[insulin["BWZ Carb Input (grams)"] > 0]

meal_matrix = []
carbs = []

for _, row in insulin.iterrows():
    t = row["datetime"]
    window = cgm[(cgm["datetime"] >= t) &
                 (cgm["datetime"] < t + pd.Timedelta(minutes=150))]

    values = window["Sensor Glucose (mg/dL)"].values

    if len(values) >= 30:
        meal_matrix.append(values[:30])
        carbs.append(row["BWZ Carb Input (grams)"])

X = np.array(meal_matrix)
carbs = np.array(carbs)

# Ground truth bins
min_carb = carbs.min()
bins = ((carbs - min_carb) // 20).astype(int)
n = bins.max() + 1

# Preprocess
X = np.nan_to_num(X)
X = StandardScaler().fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
k_labels = kmeans.labels_

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5).fit(X)
d_labels = dbscan.labels_

# SSE
def sse(X, labels):
    total = 0
    for c in set(labels):
        pts = X[labels == c]
        if len(pts) == 0:
            continue
        center = pts.mean(axis=0)
        total += ((pts - center) ** 2).sum()
    return total

# Cluster-bin matrix
def matrix(labels, bins):
    m = np.zeros((len(set(labels)), len(set(bins))))
    cl_map = {c:i for i,c in enumerate(set(labels))}
    bn_map = {b:i for i,b in enumerate(set(bins))}
    for l, b in zip(labels, bins):
        m[cl_map[l]][bn_map[b]] += 1
    return m

# Entropy & Purity
def entropy_purity(m):
    total = m.sum()
    ent = 0
    pur = 0
    for row in m:
        s = row.sum()
        if s == 0:
            continue
        p = row / s
        ent += (s/total) * entropy(p, base=2)
        pur += row.max()
    return ent, pur/total

# Compute metrics
sse_k = sse(X, k_labels)
sse_d = sse(X, d_labels)

mk = matrix(k_labels, bins)
md = matrix(d_labels, bins)

ek, pk = entropy_purity(mk)
ed, pd_ = entropy_purity(md)

# Save result
pd.DataFrame([[sse_k, sse_d, ek, ed, pk, pd_]]).to_csv(
    "Result.csv", header=False, index=False
)