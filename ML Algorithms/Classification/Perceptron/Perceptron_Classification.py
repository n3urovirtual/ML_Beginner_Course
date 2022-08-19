# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Iris_flower_dataset.csv")

# Declare feature and target variables
X = dataset.iloc[:, [0, 3]].values
Y = dataset.iloc[:, 4].values

# Convert categorical variable into numerical using Label Encoder
from sklearn import preprocessing

lab_encode = preprocessing.LabelEncoder()
Y = lab_encode.fit_transform(Y)

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import the Perceptron Classifier with parameters: 400 epochs over the data
# and 0.1 learning rate
from sklearn.linear_model import Perceptron

ppton = Perceptron(max_iter=400, eta0=0.1)

# Train the classifier
ppton.fit(X_train, Y_train)

# Predict the testing set results
Y_pred = ppton.predict(X_test)

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

# Example prediction with real values
prediction = ppton.predict([["X1", "X2"]])

# Training set results plot & visualization
from matplotlib.colors import ListedColormap

plt1 = plt.figure(1)
X1, X2 = np.meshgrid(
    np.arange(start=X_train[:, 0].min() - 1, stop=X_train[:, 0].max() + 1, step=0.01),
    np.arange(start=X_train[:, 1].min() - 1, stop=X_train[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    ppton.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
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
plt.title("Perceptron (Training set)")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Width")
plt.legend()
plt1.show()

# Testing set results plot & visualization
from matplotlib.colors import ListedColormap

plt2 = plt.figure(2)
X1, X2 = np.meshgrid(
    np.arange(start=X_test[:, 0].min() - 1, stop=X_test[:, 0].max() + 1, step=0.01),
    np.arange(start=X_test[:, 1].min() - 1, stop=X_test[:, 1].max() + 1, step=0.01),
)
plt.contourf(
    X1,
    X2,
    ppton.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
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
plt.title("Perceptron (Test set)")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Width")
plt.legend()
plt2.show()
