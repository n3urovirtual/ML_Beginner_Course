# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Wines.csv")
X = dataset.iloc[:, 0:13].values
Y = dataset.iloc[:, 13].values

# Feature scaling (necessary when using PCA or LDA)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
# X = StandardScaler().fit_transform(X)

# Import and apply PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=None)
# Fit PCA on training set only
components = pca.fit_transform(X_scaled)

# The explained variance tells us how much variance
# can be attributed to each of the principal components
variance = pca.explained_variance_ratio_

# Plot the visualisation of PCA
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(X[:, 0], c="red", marker="*")
ax.scatter(X[:, 1], c="blue", marker="^")
ax.scatter(X[:, 2], c="green", marker="o")

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
