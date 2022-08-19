# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Pima_Diabetes.csv")

# Declare independent (X) and dependent(Y) variables
X = dataset.iloc[:, 0:8].values
Y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Create the ANN model
model = Sequential()

# Add the input layer and the first hidden layer
model.add(Dense(8, input_dim=8, init="uniform", activation="relu"))

# Add the second hidden layer
model.add(Dense(12, init="uniform", activation="relu"))
model.add(Dense(6, init="uniform", activation="relu"))

# Add the output layer
model.add(Dense(1, init="uniform", activation="sigmoid"))

# Compiling the ANN model (find the best set of weights to make predictions)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit the ANN model to the training set
history = model.fit(X_train, Y_train, epochs=500, batch_size=10)

# Predict the Test set results
prediction = model.predict(X_test)
rounded_prediction = [int(round(x[0])) for x in prediction]

# Evaluate the model
score = model.evaluate(X_train, Y_train)
print("Model score:", score)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

print(classification_report(Y_test, rounded_prediction))
cm = confusion_matrix(Y_test, rounded_prediction)
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

# create a matrix for the confusion matrix
acc = metrics.accuracy_score(Y_test, rounded_prediction)
print("Accuracy of the model", round(acc * 100, 1), "%")

# Make a real prediction using a random set of numbers
random_set = [1, 85, 66, 29, 0, 26.6, 0.4, 31]
random_set = np.expand_dims(random_set, axis=0)
model.predict(random_set)
