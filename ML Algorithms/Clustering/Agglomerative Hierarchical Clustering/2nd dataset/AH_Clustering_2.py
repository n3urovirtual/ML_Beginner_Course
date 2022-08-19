# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("US_Crime.csv")
X = dataset.iloc[:, [1, 2, 4]].values

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

usrmetric = "euclidean"
usrlinkage = "ward"

# Create a dendrogram to estimate the optimal number of clusters
from scipy.cluster.hierarchy import linkage, dendrogram

sch = linkage(X, metric=usrmetric, method=usrlinkage)
dendrogram(sch)
plt1 = plt.figure(1)
plt.xlabel("Customers")
plt.ylabel("Between Centroid Euclidean Distance")
plt.title("Dendrogram")
plt1.show()

# Fit AHC to the dataset
from sklearn.cluster import AgglomerativeClustering

ahc = AgglomerativeClustering(n_clusters=4, affinity=usrmetric, linkage=usrlinkage)

# Predict the cluster for each customer
y_ahc = ahc.fit_predict(X)

# Silhouette score
from sklearn import metrics

score = metrics.silhouette_score(X, y_ahc)

# Plotting and visualizing clusters of customers
from mpl_toolkits.mplot3d import Axes3D

plt2 = plt.figure(2)
ax = plt2.add_subplot(111, projection="3d")
ax.scatter(
    X[y_ahc == 0, 0],
    X[y_ahc == 0, 1],
    X[y_ahc == 0, 2],
    s=75,
    c="blue",
    label="Cluster 1",
)
ax.scatter(
    X[y_ahc == 1, 0],
    X[y_ahc == 1, 1],
    X[y_ahc == 1, 2],
    s=75,
    c="black",
    label="Cluster 2",
)
ax.scatter(
    X[y_ahc == 2, 0],
    X[y_ahc == 2, 1],
    X[y_ahc == 2, 2],
    s=75,
    c="red",
    label="Cluster 3",
)
ax.scatter(
    X[y_ahc == 3, 0],
    X[y_ahc == 3, 1],
    X[y_ahc == 3, 2],
    s=75,
    c="yellow",
    label="Cluster 4",
)
ax.set_title("Clusters of states based on crime rate")
ax.set_xlabel("Murder")
ax.set_ylabel("Assault")
ax.set_zlabel("Rape")
plt.legend()
plt2.show()
