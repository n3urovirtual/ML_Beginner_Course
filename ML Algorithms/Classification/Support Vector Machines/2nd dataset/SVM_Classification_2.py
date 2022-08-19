# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("BankNote_Authentication.csv")

# Declare independent and dependent variables
X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values

# Split dataset into the train set & test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import Support Vector Machine model
from sklearn.svm import SVC

# Create an SVM classifier
classifier = SVC(kernel="linear", random_state=0)
# classifier =SVC(kernel = 'rbf', random_state = 0)

# Train the model using the training sets
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_predicted = classifier.predict(X_test)

# Get the Confusion Matrix and the accuracy of the model
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
print("Accuracy of the model:", round(acc * 100, 1), "%")
