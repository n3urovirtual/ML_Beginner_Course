# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("Cars.csv")

# Apply feature scaling
from sklearn.preprocessing import StandardScaler

scaledX = StandardScaler()
dataset = scaledX.fit_transform(dataset)

# Declare independent (X) and dependent(Y) variables
X = dataset[:, :-1]
Y = dataset[:, -1:]

# Import and apply PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=None)
X = pca.fit_transform(X)

# The explained variance tells us how much variance
# can be attributed to each of the principal components
variance_exp = pca.explained_variance_ratio_

# A kind contribution by Damian Zięba
objects = ("PC1", "PC2", "PC3", "PC4", "PC5")
xticks = np.arange(len(objects))

plt0 = plt.figure(0)
plt.bar(xticks, variance_exp, align="center")
plt.xticks(xticks, objects)
plt.ylabel("Variance")
plt.title("Variance of components")
plt0.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

# Import Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Create the ANN model
model = Sequential()

# Add the input layer and the first hidden layer
model.add(Dense(12, input_dim=5, kernel_initializer="normal", activation="relu"))

# Add the second hidden layer
model.add(Dense(8, activation="relu"))

# Add output layer
model.add(Dense(1, activation="linear"))

# Compiling the ANN model (find the best set of weights to make predictions)
model.compile(
    optimizer="adam", loss="mse", metrics=["mean_absolute_error", "mean_squared_error"]
)

# Fit the ANN model to the training set
history = model.fit(X_train, Y_train, epochs=500, batch_size=10)

# Predict the Test set results
prediction = model.predict(X_test)

# Evaluation of the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(Y_test, prediction)
variance = np.var(prediction)
r2 = r2_score(Y_test, prediction)
