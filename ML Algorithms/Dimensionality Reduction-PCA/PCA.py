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

scaledX = StandardScaler()
X = scaledX.fit_transform(X)

# Import and apply PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=None)
X = pca.fit_transform(X)

# The explained variance tells us how much variance
# can be attributed to each of the principal components
variance_exp = pca.explained_variance_ratio_

# Select only the components you need
X = X[:, 0:3]

# Spit data into Training and Test Sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4)

# Fit Logistic Regression to the Training Set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
probability = classifier.predict_proba(X_test)

# Predict Test set results
Y_pred = classifier.predict(X_test)

# Make real life prediction
prediction = classifier.predict([[X1, X2, X3]])

# Evaluation of the model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

print(classification_report(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)
TP_0 = cm[0][0]
TP_1 = cm[1][1]
TP_2 = cm[2][2]
TN_0 = TP_1 + TP_2
TN_1 = TP_0 + TP_2
TN_2 = TP_0 + TP_1
FP_0 = cm[[1, 2], 0]
FP_1 = cm[[0, 2], 1]
FP_2 = cm[[0, 1], 2]
FN_0 = cm[0, [1, 2]]
FN_1 = cm[1, [0, 2]]
FN_2 = cm[2, [0, 1]]
values = [
    [(TP_0), (TN_0), (FP_0), (FN_0)],
    [(TP_1), (TN_1), (FP_1), (FN_1)],
    [(TP_2), (TN_2), (FP_2), (FN_2)],
]
c_matrix = pd.DataFrame(
    values, columns=["TP", "TN", "FP", "FN"], index=["class 0", "class 1", "class 2"]
)
print(c_matrix)
acc = metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy of the model", round(acc * 100, 1), "%")

# Plot and visualize the Training set results
from mpl_toolkits.mplot3d import Axes3D

plt1 = plt.figure(1)
ax = plt1.add_subplot(111, projection="3d")
ax.scatter(
    X_train[Y_train == 1, 0],
    X_train[Y_train == 1, 1],
    X_train[Y_train == 1, 2],
    c="red",
    marker="*",
)
ax.scatter(
    X_train[Y_train == 2, 0],
    X_train[Y_train == 2, 1],
    X_train[Y_train == 2, 2],
    c="blue",
    marker="^",
)
ax.scatter(
    X_train[Y_train == 3, 0],
    X_train[Y_train == 3, 1],
    X_train[Y_train == 3, 2],
    c="green",
    marker="o",
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt1.show()

# Plot and visualize the Test set results
from mpl_toolkits.mplot3d import Axes3D

plt2 = plt.figure(2)
ax = plt2.add_subplot(111, projection="3d")
ax.scatter(
    X_test[Y_test == 1, 0],
    X_test[Y_test == 1, 1],
    X_test[Y_test == 1, 2],
    c="red",
    marker="*",
)
ax.scatter(
    X_test[Y_test == 2, 0],
    X_test[Y_test == 2, 1],
    X_test[Y_test == 2, 2],
    c="blue",
    marker="^",
)
ax.scatter(
    X_test[Y_test == 3, 0],
    X_test[Y_test == 3, 1],
    X_test[Y_test == 3, 2],
    c="green",
    marker="o",
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt2.show()
