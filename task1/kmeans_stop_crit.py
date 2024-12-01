import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import normalize
from collections import Counter

# Load data
data = pd.read_csv("kmeans_data/data.csv", header=None).values
labels = pd.read_csv("kmeans_data/label.csv", header=None).values.flatten()
data = normalize(data)

# Distance Functions
def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_distance(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return 1 - dot_product / (norm_x * norm_y)

def generalized_jaccard_distance(x, y):
    min_sum = np.sum(np.minimum(x, y))
    max_sum = np.sum(np.maximum(x, y))
    return 1 - (min_sum / max_sum)

# K-means Algorithm
def kmeans(X, k, distance_metric, max_iter=100, tol=1e-4, stopping_criterion="centroid"):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    prev_centroids = np.zeros_like(centroids)
    labels = np.zeros(n_samples)
    sse_history = []
    start_time = time.time()

    for iteration in range(max_iter):
        # Assignment Step
        for i, x in enumerate(X):
            distances = [distance_metric(x, centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)

        # Compute SSE
        sse = sum([np.sum([distance_metric(x, centroids[int(labels[i])]) ** 2 for i, x in enumerate(X)])])
        sse_history.append(sse)

        # Update Step
        prev_centroids = centroids.copy()
        for j in range(k):
            points = X[labels == j]
            if points.size:
                centroids[j] = np.mean(points, axis=0)

        # Convergence Conditions
        if stopping_criterion == "centroid" and np.linalg.norm(centroids - prev_centroids) < tol:
            break
        if stopping_criterion == "sse_increase" and iteration > 0 and sse_history[-1] > sse_history[-2]:
            break
        if stopping_criterion == "max_iter" and iteration >= max_iter - 1:
            break

    end_time = time.time()
    return centroids, labels, sse_history, iteration + 1, end_time - start_time

# Store results in a text file
def save_results_to_file(results, filename):
    with open(filename, "w") as f:
        for metric, result in results.items():
            f.write(f"Results for {metric} Distance:\n")
            f.write(f"  Stopping Criterion: {result['Stopping Criterion']}\n")
            f.write(f"  SSE: {result['SSE']:.4f}\n")
            f.write(f"  Iterations: {result['Iterations']}\n")
            f.write(f"  Time Taken: {result['Time']:.2f} seconds\n\n")

# Run K-means for each metric and stopping criterion
def run_kmeans_for_criteria(data, labels, stopping_criterion, filename):
    metrics = {
        "Euclidean": euclidean_distance,
        "Cosine": cosine_distance,
        "Jaccard": generalized_jaccard_distance
    }

    k = 10
    results = {}
    for metric_name, metric_func in metrics.items():
        centroids, predicted_labels, sse_history, iterations, elapsed_time = kmeans(
            data, k, metric_func, max_iter=100, tol=1e-4, stopping_criterion=stopping_criterion
        )
        results[metric_name] = {
            "Stopping Criterion": stopping_criterion,
            "SSE": sse_history[-1],
            "Iterations": iterations,
            "Time": elapsed_time
        }

    save_results_to_file(results, filename)
    print(f"Results saved to {filename}")

# Example usage for three stopping criteria
run_kmeans_for_criteria(data, labels, "centroid", "results_centroid.txt")
run_kmeans_for_criteria(data, labels, "sse_increase", "results_sse_increase.txt")
run_kmeans_for_criteria(data, labels, "max_iter", "results_max_iter.txt")
