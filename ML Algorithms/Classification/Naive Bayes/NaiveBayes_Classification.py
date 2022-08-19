# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

# Split into the train set & test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Create a classifier
classifier = GaussianNB()

# Train the model using the training sets
classifier.fit(X_train, Y_train)

# Calculate the probability for each class using X_test
probability = classifier.predict_proba(X_test)

# Predict the testing set output
Y_predicted = classifier.predict(X_test)
example_prediction = classifier.predict([[-0.2, 3]])
print("Predicted Value:", example_prediction)

# Evaluation of the model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

print(classification_report(Y_test, Y_predicted))
cm = confusion_matrix(Y_test, Y_predicted)
true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_positive = cm[1][1]
values = [
    [("true negative:", true_negative), ("false positive:", false_positive)],
    [("false negative:", false_negative), ("true positive:", true_positive)],
]
c_matrix = pd.DataFrame(
    values, columns=["Negative", "Positive"], index=["Negative", "Positive"]
)
print(c_matrix)
acc = metrics.accuracy_score(Y_test, Y_predicted)
print("Accuracy of the model", round(acc * 100, 1), "%")

# Training set results plot & visualization
from matplotlib.colors import ListedColormap

plt1 = plt.figure(1)
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.5,
    cmap=ListedColormap(("Grey", "White")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("black", "red"))(i),
        label=j,
    )
plt.title("Naive Bayes (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt1.show()

# Testing set results plot & visualization
from matplotlib.colors import ListedColormap

plt2 = plt.figure(2)
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.5,
    cmap=ListedColormap(("Grey", "White")),
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(("black", "red"))(i),
        label=j,
    )
plt.title("Naive Bayes (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt2.show()
