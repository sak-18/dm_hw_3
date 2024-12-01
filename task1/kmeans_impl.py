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

# K-means Algorithm with SSE and Convergence Tracking
def kmeans(X, k, distance_metric, max_iter=100, tol=1e-4, track_time=False):
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    prev_centroids = np.zeros_like(centroids)
    labels = np.zeros(n_samples)
    sse_history = []
    start_time = time.time()

    print(f"Running K-means with k={k}, max_iter={max_iter}, and {distance_metric.__name__} distance...")
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}...")

        # Assignment Step
        for i, x in enumerate(X):
            distances = [distance_metric(x, centroid) for centroid in centroids]
            labels[i] = np.argmin(distances)
        print("  Assignment step completed.")

        # Compute SSE
        sse = sum([np.sum([distance_metric(x, centroids[int(labels[i])]) ** 2 for i, x in enumerate(X)])])
        sse_history.append(sse)
        print(f"  SSE after iteration {iteration + 1}: {sse:.4f}")

        # Update Step
        prev_centroids = centroids.copy()
        for j in range(k):
            points = X[labels == j]
            if points.size:
                centroids[j] = np.mean(points, axis=0)
        print("  Update step completed.")

        # Convergence Conditions
        if np.linalg.norm(centroids - prev_centroids) < tol:
            print(f"  Converged due to centroid position change within tolerance ({tol}).")
            break
        if iteration > 0 and sse_history[-1] > sse_history[-2]:
            print("  Converged due to SSE increase.")
            break

    end_time = time.time() if track_time else None
    print(f"K-means completed in {iteration + 1} iterations.\n")
    return centroids, labels, sse_history, iteration + 1, end_time - start_time if track_time else None

# Run K-means with metrics
metrics = {
    "Euclidean": euclidean_distance,
    "Cosine": cosine_distance,
    "Jaccard": generalized_jaccard_distance
}

k = 10
results = {}
for metric_name, metric_func in metrics.items():
    print(f"Starting clustering with {metric_name} distance...")
    centroids, predicted_labels, sse_history, iterations, elapsed_time = kmeans(data, k, metric_func, max_iter=500, track_time=True)

    # Assign labels using majority vote
    print("Assigning majority-vote labels...")
    cluster_to_label = {}
    for cluster in range(k):
        cluster_points = labels[predicted_labels == cluster]
        if len(cluster_points) > 0:
            cluster_to_label[cluster] = Counter(cluster_points).most_common(1)[0][0]
    predicted_labels = np.array([cluster_to_label[label] for label in predicted_labels])
    print("Label assignment completed.")

    # Compute metrics
    accuracy = np.mean(predicted_labels == labels)
    sse = sse_history[-1]
    results[metric_name] = {"SSE": sse, "Accuracy": accuracy, "Iterations": iterations, "Time": elapsed_time}
    print(f"Results for {metric_name} distance: SSE={sse:.4f}, Accuracy={accuracy:.4f}, Iterations={iterations}, Time={elapsed_time:.2f}s\n")

# Final results
for metric, result in results.items():
    print(f"Summary for {metric} Distance:")
    print(f"  SSE: {result['SSE']:.4f}")
    print(f"  Accuracy: {result['Accuracy']:.4f}")
    print(f"  Iterations: {result['Iterations']}")
    print(f"  Time Taken: {result['Time']:.2f} seconds\n")
