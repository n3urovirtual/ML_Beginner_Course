# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import dataset
dataset = pd.read_csv("Diabetes.csv")

# Visualize classes of diagnosis column
plt.figure(figsize=(8, 4))
sns.countplot(dataset["Outcome"], palette="coolwarm")

# Generate and visualize the correlation matrix
corr = dataset.corr().round(1)
# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set figure size
f, ax = plt.subplots(figsize=(20, 20))
# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    annot=True,
)
plt.tight_layout()

# Declare the variables
X = dataset.iloc[:, 0:8].values
Y = dataset.iloc[:, 8].values

# Split the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
probability = log_reg.predict_proba(X_test)

# Predict test set results
Y_predicted = log_reg.predict(X_test)

# Evaluation of the model
from sklearn.metrics import confusion_matrix, classification_report

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
print(
    "Correct Predictions",
    round((true_negative + true_positive) / len(Y_predicted) * 100, 1),
    "%",
)
