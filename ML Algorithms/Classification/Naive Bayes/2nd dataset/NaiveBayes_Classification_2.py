# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv("seeds.csv")
X = dataset.iloc[:, 0:7].values
Y = dataset.iloc[:, 7].values

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

# Predict the testing set output
Y_predicted = classifier.predict(X_test)

# Evaluation of the model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

print(classification_report(Y_test, Y_predicted))
cm = confusion_matrix(Y_test, Y_predicted)
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
acc = metrics.accuracy_score(Y_test, Y_predicted)
print("Accuracy of the model", round(acc * 100, 1), "%")
