# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")

# Declare variables
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

labelencoder_Geog = LabelEncoder()
X[:, 1] = labelencoder_Geog.fit_transform(X[:, 1])
labelencoder_Gend = LabelEncoder()
X[:, 2] = labelencoder_Gend.fit_transform(X[:, 2])

# Create dummy variables for  'Geography'
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

# Avoid dummy variable trap
X = X[:, 1:]  # take all the columns except the first one

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(input_dim=11, output_dim=6, init="uniform", activation="relu"))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))

# Adding the output layer
classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, epochs=500, batch_size=10)


# Predicting the Test set results
Y_pred = classifier.predict(X_test)
rounded_prediction = [int(round(x[0])) for x in Y_pred]

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
