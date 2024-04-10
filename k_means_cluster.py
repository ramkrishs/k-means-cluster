import numpy as np
import matplotlib.pyplot as plt

class KMeansCluster:
    def __init__(self):
        pass

    def fit(self, X, K, max_iters=100):
        """
        Fit K-means clustering algorithm to the data.

        Args:
            X (ndarray): Input data points of shape (m, n).
            K (int): Number of clusters.
            max_iters (int): Maximum number of iterations. Defaults to 100.

        Returns:
            centroids (ndarray): Final centroids of shape (K, n).
            idx (ndarray): Indices of the closest centroids for each data point of shape (m,).
        """
        m, n = X.shape
        centroids = self._initialize_centroids(X, K)
        prev_centroids = centroids
        idx = np.zeros(m)

        for i in range(max_iters):
            print("K-Means iteration %d/%d" % (i, max_iters - 1))
            idx = self._assign_to_closest_centroids(X, centroids)
            centroids = self._update_centroids(X, idx, K)
            if np.all(centroids == prev_centroids):
                break
            prev_centroids = centroids

        return centroids, idx

    def _initialize_centroids(self, X, K):
        """
        Initialize centroids randomly.

        Args:
            X (ndarray): Input data points.
            K (int): Number of centroids.

        Returns:
            centroids (ndarray): Initialized centroids.
        """
        randidx = np.random.permutation(X.shape[0])
        centroids = X[randidx[:K]]
        return centroids

    def _assign_to_closest_centroids(self, X, centroids):
        """
        Assign each data point to the closest centroid.

        Args:
            X (ndarray): Input data points.
            centroids (ndarray): Centroids.

        Returns:
            idx (ndarray): Indices of the closest centroids for each data point.
        """
        idx = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            idx[i] = np.argmin(distances)

        return idx

    def _update_centroids(self, X, idx, K):
        """
        Update centroids based on the mean of data points assigned to each centroid.

        Args:
            X (ndarray): Input data points.
            idx (ndarray): Indices of the closest centroids for each data point.
            K (int): Number of centroids.

        Returns:
            centroids (ndarray): Updated centroids.
        """
        centroids = np.zeros((K, X.shape[1]))

        for k in range(K):
            points = X[idx == k]
            centroids[k] = np.mean(points, axis=0)

        return centroids


if __name__ == "__main__":
    X = np.load("data/data.npy")
    K = 3
    kmeans = KMeansCluster()
    centroids, idx = kmeans.fit(X=X, K=K)

    print("Final Centroids:")
    print(centroids)
    print("Indices of Closest Centroids:")
    print(idx)
