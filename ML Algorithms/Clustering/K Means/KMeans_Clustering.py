# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Using the elbow method to determine the optimal number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Create a plot in order to observe the elbow
plt1 = plt.figure(1)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")  # Within Clusters Sum of Squares
plt1.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10)

# Predict the cluster for each data point
y_kmeans = kmeans.fit_predict(X)

# Example prediction
prediction = kmeans.predict([[70, 40]])
"""to find the correct predicted cluster add 1 because clusters numbering
in python starts from 0"""

# Make centroids for each cluster
centroids = kmeans.cluster_centers_

# Silhouette score
"""Silhouette coefficients (as these values are referred to as) near +1 indicate that 
the sample is far away from the neighboring clusters. A value of 0 indicates that the 
sample is on or very close to the decision boundary between two neighboring 
clusters and negative values indicate that those samples might have been assigned to the wrong cluster."""
from sklearn import metrics

score = metrics.silhouette_score(X, y_kmeans)

# Instantiate Silhouette Visualizer
from yellowbrick.cluster import SilhouetteVisualizer

plt2 = plt.figure(2)
visualizer = SilhouetteVisualizer(KMeans(5))
visualizer.fit(X)  # Fit the data to the visualizer
visualizer.poof()
plt2.show

# Visualising the clusters
plt3 = plt.figure(3)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c="blue", label="Cluster 1")
plt.scatter(
    X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c="black", label="Cluster 2"
)
plt.scatter(
    X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c="orange", label="Cluster 3"
)
plt.scatter(
    X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c="yellow", label="Cluster 4"
)
plt.scatter(
    X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c="purple", label="Cluster 5"
)
plt.scatter(
    centroids[:, 0], centroids[:, 1], marker="*", s=300, c="red", label="Centroids"
)
plt.title("Clusters of Customers")
plt.xlabel("Annual Income in $")
plt.ylabel("Spending Score")
plt.legend()
plt3.show()
