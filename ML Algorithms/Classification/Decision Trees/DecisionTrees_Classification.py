# Import libraries
import numpy as np
import pandas as pd

# Import dataset
dataset = pd.read_csv("Pima_diabetes.csv")

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

# Fit Decision Tree Classifier to the training set
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3)
classifier.fit(X_train, Y_train)
probability = classifier.predict_proba(X_test)

# Predict test set results
Y_pred = classifier.predict(X_test)

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

# Create a visualization for the decision tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(
    classifier,
    out_file=dot_data,
    filled=True,
    rounded=True,
    precision=2,
    special_characters=True,
    feature_names=dataset.columns.drop("Outcome"),
    class_names=["0", "1"],
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("Diabetes Decision Tree.png")
Image(graph.create_png())
