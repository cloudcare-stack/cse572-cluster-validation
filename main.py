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
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.stats import entropy as scipy_entropy

# File paths
RESULT_FILE = "Result.csv"

# Function definitions
def load_data():
    # Load CGM and insulin data from CSV files, combine date and time into datetime, and sort by datetime.
    cgm = pd.read_csv("CGMData.csv", low_memory=False)
    insulin = pd.read_csv("InsulinData.csv", low_memory=False)

    # Combine date and time into a single datetime column for both datasets, handling errors gracefully.
    cgm["datetime"] = pd.to_datetime(
        cgm["Date"].astype(str) + " " + cgm["Time"].astype(str),
        errors="coerce"
    )

    # Combine date and time into a single datetime column for insulin data, handling errors gracefully.
    insulin["datetime"] = pd.to_datetime(
        insulin["Date"].astype(str) + " " + insulin["Time"].astype(str),
        errors="coerce"
    )

    # Drop rows with invalid datetime values and sort the data by datetime.
    cgm = cgm.dropna(subset=["datetime"])
    insulin = insulin.dropna(subset=["datetime"])

    # Sort both datasets by datetime to ensure proper temporal alignment for subsequent analysis.
    cgm = cgm.sort_values("datetime")
    insulin = insulin.sort_values("datetime")

    return cgm, insulin

# Let’s create some functions that can figure out which columns to use for glucose and carbohydrate input, even if there are different names for them.
def get_sensor_column(cgm):
    # Check for common column names that might represent the CGM glucose values, and return the first match found.
    possible_cols = [
        "Sensor Glucose (mg/dL)",
        "Sensor Glucose",
        "Glucose Sensor Value",
        "Glucose"
    ]

    # Iterate through the list of possible column names and return the first one that exists in the CGM dataset.
    for col in possible_cols:
        if col in cgm.columns:
            return col

    raise ValueError("Could not find CGM glucose column.")

# Check for common column names that might represent the insulin carbohydrate input, and return the first match found.
def get_carb_column(insulin):
    # Define a list of possible column names that could represent the carbohydrate input in the insulin dataset, and return the first match found.
    possible_cols = [
        "BWZ Carb Input (grams)",
        "BWZ Carb Input",
        "Carb Input",
        "Carbs"
    ]

    # Iterate through the list of possible column names and return the first one that exists in the insulin dataset.
    for col in possible_cols:
        if col in insulin.columns:
            return col

    raise ValueError("Could not find insulin carb input column.")

# Let’s gather the glucose data from meals, focusing on how much insulin is used to manage the carbs. Then, we’ll put it all together in a matrix, showing the glucose levels after each meal and what carbs were involved.
def extract_meal_data(cgm, insulin):
    # Identify the correct columns for glucose values in the CGM dataset and carbohydrate input in the insulin dataset, handling multiple possible column names.
    sensor_col = get_sensor_column(cgm)
    carb_col = get_carb_column(insulin)

    # Convert the carbohydrate input column to numeric values, coercing errors to NaN, to ensure proper handling of non-numeric entries.
    insulin[carb_col] = pd.to_numeric(insulin[carb_col], errors="coerce")

    # Identify meal events in the insulin dataset where the carbohydrate input is greater than zero, and create a DataFrame containing the datetime and carbohydrate amount for each meal event.
    meal_events = insulin[
        (insulin[carb_col].notna()) &
        (insulin[carb_col] > 0)
    ][["datetime", carb_col]]

    # Initialize empty lists to store the meal-related glucose data and corresponding carbohydrate labels, which will be populated in the following loop.
    meal_matrix = []
    carb_labels = []

    # Iterate through each meal event, extracting the glucose values from the CGM dataset for a 150-minute window following the meal time, and store the glucose values and corresponding carbohydrate amount in the meal matrix and carb labels lists.
    for _, row in meal_events.iterrows():
        meal_time = row["datetime"]
        carb_amount = row[carb_col]

        start_time = meal_time
        end_time = meal_time + pd.Timedelta(minutes=150)

        window = cgm[
            (cgm["datetime"] >= start_time) &
            (cgm["datetime"] < end_time)
        ].sort_values("datetime")

        glucose_values = pd.to_numeric(window[sensor_col], errors="coerce").values

        if len(glucose_values) >= 30:
            glucose_values = glucose_values[:30]

            if np.isnan(glucose_values).all():
                continue

            meal_matrix.append(glucose_values)
            carb_labels.append(carb_amount)

    meal_matrix = np.array(meal_matrix, dtype=float)
    carb_labels = np.array(carb_labels, dtype=float)

    return meal_matrix, carb_labels

# Now, we need to create bins for the carbohydrate intake to use as ground truth labels for our clustering evaluation. We’ll group the carb amounts into 20-gram intervals and assign each meal to a corresponding bin.
def make_ground_truth_bins(carb_labels):
    min_carb = np.nanmin(carb_labels)
    max_carb = np.nanmax(carb_labels)

    bin_labels = np.floor((carb_labels - min_carb) / 20).astype(int)

    number_of_bins = int(np.floor((max_carb - min_carb) / 20)) + 1

    bin_labels = np.clip(bin_labels, 0, number_of_bins - 1)

    return bin_labels, number_of_bins

# Before we can run our clustering algorithms, we need to preprocess the meal matrix. This involves handling any missing values by imputing them with the mean of each feature, and then standardizing the features to ensure they are on the same scale for better clustering performance.
def preprocess_features(meal_matrix):
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(meal_matrix)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

# To evaluate the quality of our clusters, we’ll compute the Sum of Squared Errors (SSE) for each clustering result. This metric will help us understand how well the clusters are formed by measuring the total squared distance between each point and its cluster center.
def compute_sse(X, cluster_labels):
    total_sse = 0.0

    for cluster in np.unique(cluster_labels):
        cluster_points = X[cluster_labels == cluster]

        if len(cluster_points) == 0:
            continue

        center = np.mean(cluster_points, axis=0)
        total_sse += np.sum((cluster_points - center) ** 2)

    return float(total_sse)

# Next, we need to create a matrix that counts how many meals from each carbohydrate bin fall into each cluster. This will allow us to compute the entropy and purity of our clusters based on the distribution of the ground truth bins within each cluster.
def make_cluster_bin_matrix(cluster_labels, bin_labels):
    unique_clusters = np.unique(cluster_labels)
    unique_bins = np.unique(bin_labels)

    matrix = np.zeros((len(unique_clusters), len(unique_bins)))

    cluster_to_row = {cluster: i for i, cluster in enumerate(unique_clusters)}
    bin_to_col = {bin_label: i for i, bin_label in enumerate(unique_bins)}

    for c, b in zip(cluster_labels, bin_labels):
        matrix[cluster_to_row[c], bin_to_col[b]] += 1

    return matrix

# Finally, we can compute the entropy and purity of our clusters based on the cluster-bin matrix. The entropy will measure the disorder or randomness of the distribution of bins within each cluster, while the purity will measure the extent to which each cluster contains data points from a single bin.
def compute_entropy_purity(cluster_bin_matrix):
    total_points = np.sum(cluster_bin_matrix)

    if total_points == 0:
        return 0.0, 0.0

    total_entropy = 0.0
    total_purity = 0.0

    for row in cluster_bin_matrix:
        cluster_total = np.sum(row)

        if cluster_total == 0:
            continue

        probabilities = row / cluster_total
        cluster_entropy = scipy_entropy(probabilities, base=2)

        total_entropy += (cluster_total / total_points) * cluster_entropy
        total_purity += np.max(row)

    purity = total_purity / total_points

    return float(total_entropy), float(purity)

# Now we can run our clustering algorithms. We’ll use KMeans to cluster the data into the same number of clusters as our ground truth bins, and we’ll use DBSCAN to find clusters based on density. For DBSCAN, we’ll perform a grid search over a range of epsilon and minimum samples values to find the best clustering result based on the number of clusters and noise points.
def run_kmeans(X, n_clusters):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)
    return labels

# For DBSCAN, we’ll reduce the dimensionality of the data using PCA to improve clustering performance, and then we’ll iterate over a range of epsilon and minimum samples values to find the best clustering result based on a custom scoring function that considers both the number of clusters and the amount of noise.
def run_dbscan(X):
    pca = PCA(n_components=min(5, X.shape[1]))
    X_reduced = pca.fit_transform(X)

    best_labels = None
    best_score = -1

    eps_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    min_samples_values = [3, 4, 5, 6, 8]

    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_reduced)

            number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = np.sum(labels == -1)

            if number_of_clusters <= 1:
                continue

            score = number_of_clusters - (noise_count / len(labels))

            if score > best_score:
                best_score = score
                best_labels = labels

    if best_labels is None:
        best_labels = DBSCAN(eps=2.0, min_samples=5).fit_predict(X_reduced)

    return best_labels

# Now we can put everything together in the main function, which will orchestrate the entire workflow from loading the data to computing the evaluation metrics and saving the results to a CSV file.
def main():
    cgm, insulin = load_data()

    meal_matrix, carb_labels = extract_meal_data(cgm, insulin)

    bin_labels, number_of_bins = make_ground_truth_bins(carb_labels)

    X = preprocess_features(meal_matrix)

    kmeans_labels = run_kmeans(X, number_of_bins)
    dbscan_labels = run_dbscan(X)

    sse_kmeans = compute_sse(X, kmeans_labels)
    sse_dbscan = compute_sse(X, dbscan_labels)

    kmeans_matrix = make_cluster_bin_matrix(kmeans_labels, bin_labels)
    dbscan_matrix = make_cluster_bin_matrix(dbscan_labels, bin_labels)

    entropy_kmeans, purity_kmeans = compute_entropy_purity(kmeans_matrix)
    entropy_dbscan, purity_dbscan = compute_entropy_purity(dbscan_matrix)

    result = np.array([[
        sse_kmeans,
        sse_dbscan,
        entropy_kmeans,
        entropy_dbscan,
        purity_kmeans,
        purity_dbscan
    ]])

    pd.DataFrame(result).to_csv(RESULT_FILE, header=False, index=False)


if __name__ == "__main__":
    main()