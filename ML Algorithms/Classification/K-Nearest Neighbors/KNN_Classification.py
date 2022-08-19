# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Iris_flower_dataset.csv")

# Declare feature and target variables
X = dataset.iloc[:, [1, 3]].values
Y = dataset.iloc[:, 4].values

# Convert categorical column (species) into numerical column using Label Encoder
from sklearn import preprocessing

lab_encode = preprocessing.LabelEncoder()
Y = lab_encode.fit_transform(Y)

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# import the K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Run with k=1-15 and check accuracy score
k_range = range(1, 16)
scores = {}
scores_list = []
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    scores[k] = metrics.accuracy_score(Y_test, Y_pred)
    scores_list.append(metrics.accuracy_score(Y_test, Y_pred))
    print(scores)

# Visualize the relationship between K and the testing accuracy
plt1 = plt.figure(1)
plt.plot(k_range, scores_list)
plt.title("Relationship between K neighbors and Testing Accuracy")
plt.xlabel("Value of K")
plt.ylabel("Accuracy score")
plt1.show()


# Create KNN classifier with 3 neighbors and standard euclidean distance
classifier = KNeighborsClassifier(n_neighbors="see plot before")
# Train the model using the training sets
classifier.fit(X_train, Y_train)

# Predict the output from test dataset
Y_pred = classifier.predict(X_test)

# Evaluation of the model
from sklearn.metrics import confusion_matrix, classification_report

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


# Real prediction example
test = [["X1", "X2"]]
real_pred = classifier.predict(test)

# Training set results plot & visualization
from matplotlib.colors import ListedColormap

plt2 = plt.figure(2)
X1, X2 = np.meshgrid(
    np.arange(start=X_train[:, 0].min() - 1, stop=X_train[:, 0].max() + 1, step=0.01),
    np.arange(start=X_train[:, 1].min() - 1, stop=X_train[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.4,
    cmap=ListedColormap(("Cyan", "Green", "Red")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_train)):
    plt.scatter(
        X_train[Y_train == j, 0],
        X_train[Y_train == j, 1],
        c=ListedColormap(("blue", "green", "red"))(i),
        label=j,
    )
plt.title("K-Nearest Neighbors (Training set)")
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()
plt2.show()

# Testing set results plot & visualization
from matplotlib.colors import ListedColormap

plt3 = plt.figure(3)
X1, X2 = np.meshgrid(
    np.arange(start=X_test[:, 0].min() - 1, stop=X_test[:, 0].max() + 1, step=0.01),
    np.arange(start=X_test[:, 1].min() - 1, stop=X_test[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.4,
    cmap=ListedColormap(("Cyan", "Green", "Red")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_test)):
    plt.scatter(
        X_test[Y_test == j, 0],
        X_test[Y_test == j, 1],
        c=ListedColormap(("blue", "green", "red"))(i),
        label=j,
    )
plt.title("K-Nearest Neighbors (Test set)")
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()
plt3.show()
