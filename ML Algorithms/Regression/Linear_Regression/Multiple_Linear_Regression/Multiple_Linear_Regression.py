# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the dataset
dataset = pd.read_csv("Boston_housing.csv")

# Calculate pair-wise corellations for all columns
correlation_matrix = dataset.corr().round(2)

# Use seaborn's heatmap function to visualize the corellation matrix
sns.heatmap(data=correlation_matrix, annot=True, cmap="Blues")

"""From the corellation matrix we see that RM(average number of rooms) and 
LSTAT (% lower status population) are correlated (above 0.5 or -0.5) with the target 
variable MEDV (Median value of homes in $1000's)"""

# Declare predictors and outcome variable
X = dataset.iloc[:, [5, 10, 12]].values
Y = dataset.iloc[:, -1].values

# Split the dataset into the Training and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# Apply feature Scaling
from sklearn.preprocessing import StandardScaler

scaledX = StandardScaler()
X_train = scaledX.fit_transform(X_train)
X_test = scaledX.transform(X_test)

# Fit Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the Test set results
Y_predicted = regressor.predict(X_test)

# Compare the actual output values for X_test with the predicted values
comparison = pd.DataFrame(
    {"Actual Price": Y_test.flatten(), "Predicted Price": Y_predicted.flatten()}
)

# Evaluate the performance of the algorithm
from sklearn.metrics import mean_squared_error, r2_score

coef = regressor.coef_
intercept = regressor.intercept_
mse = mean_squared_error(Y_test, Y_predicted)
variance = np.var(Y_predicted)
r2 = r2_score(Y_test, Y_predicted)

# Make a real life prediction using sklearn
rm = 6
ptratio = 18
lstat = 12
set = scaledX.transform([[rm, ptratio, lstat]])
prediction = regressor.predict(set)

# Visualise actual VS predicted results
plt2 = plt.figure(2)
plt.scatter(Y_test, Y_predicted)
plt.title("Actual VS Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt2.show()


"""Alternative regression method"""
# Regression Summary using statsmodels
import statsmodels.api as sm

X = sm.add_constant(X)  # adding a constant
model = sm.OLS(Y_train, X_train).fit()
predictions = model.predict(X_test)
print_model = model.summary()
print(print_model)
mse2 = mean_squared_error(Y_test, predictions)
variance2 = np.var(predictions)
