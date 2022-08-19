# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Transfusion.csv")

# Declare feature and target variables
X = dataset.iloc[:, 0:4].values
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

# Import the Perceptron Classifier with parameters: 40 epochs over the data
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
acc = metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy of the model", round(acc * 100, 1), "%")

# Example prediction with real values
prediction = ppton.predict([["X1", "X2", "X3", "X4"]])
