# Introduction to Python

"""Variables (int:integer, str:string, bool: boolean, float:float)
Each variable represent a single value"""

# Create variables and perform basic math functions
a = 15
b = 5
print(a + b)
print(a - b)
print(a / b)
print(a * b)
print(a ** 2 + b)

# Create a height and weight variable and assign them values
height = 1.80
weight = 90

# Create and calculate bmi variable
body_mass_index = weight / height ** 2

print(body_mass_index)

# Create an intiger variable and assign it a value
hours_per_day = 24

# Create a string variable and assign it a value
last_name = "Jordan"

# Create a boolean, it can be either True or False
sun = True

"""Python Lists: a collection of many data points"""
# List x=[a,b,c,d,....]-any data type
family_age = [29, 27, 54, 50, 38]

# How to access items in a List, index starts from 0
# Given the family_age variable, 0->29, 1->27, 2->54, 3->50, 4->38
family_age[2]

# In case you have a huge list, there is also negative indexing
# which starts from the end and takes the value of -1.
# Given the family_age, -1 ->38, -2 ->50, -3 ->54, -4 ->27, -5 ->29
family_age[-4]

# List slicing. Upper limit (e.g. number 3) is not included
family_age[0:3]

family_age[:3]  # 2nd way, same result

family_age[3:]  # get list items from index 3 till the end

# A list which contains sublists
name_age = [["Chris", 29], ["Bill", 27], ["Kate", 50], ["Matthew", 54]]


"""Python Functions: pieces of reusable code to simplify things 
e.g. max(), round(), type()"""
family_age = [29, 27, 54, 50, 38, 48, 22, 36]

# Function max()
max(family_age)

# Assign the result of max() to a new variable
older = max(family_age)

type(family_age)

score = 50.23

round(score, 1)


"""Python packages/libraries and shortcuts to use them faster
Python libraries contain built in functions, methods and scripts"""
# Numerical Python: Package used to store multidimensional arrays and matrices
import numpy as np

# Pandas: Library for data manipulation and analysis
import pandas as pd

# Matplotlib: Package used to make data visualizations and plots
import matplotlib.pyplot as plt

# Scikit-learn: Useful Machine Learning Library
from sklearn.linear_model import LinearRegression

# Seaborn: Statistical data visualization
import seaborn as sns

"""Creating plots and visualise data"""
# Create a line plot using matplotlib
import matplotlib.pyplot as plt

days = [1, 2, 3, 4, 5, 6, 7]
temperature = [22, 19, 17, 16, 21, 24, 25]
plt.plot(days, temperature, linewidth=2, color="red")
plt.xlabel("Days of the week")
plt.ylabel("Temperature in Celsius")
plt.title("Weather Forecast")
plt.show()

"""Import and load a dataset (of csv,excel, json etc. format ) using Pandas"""
# CAUTION:Make sure the dataset you are trying to import is in your working directory(path)
#'Name of your variable'= pd.read_csv('Exact name of your dataset.csv')
import pandas as pd

my_dataset = pd.read_csv("Social_Network_Ads.csv")
print(my_dataset)

# Load standard datasets using sklearn
from sklearn.datasets import load_boston

dataset = load_boston()
print(dataset)


# Pandas  data selection using .iloc & .loc for rows and columns
#'name of variable'= 'name of dataset'.iloc ['row selection, 'column selection'].values

# : --> all rows selected, 3-->column with index 3
X = my_dataset.iloc[:, 3].values

# : --> all rows selected, :-1 -->all columns selected expect the last one
Y = my_dataset.iloc[:, :-1].values

# [5,10,15]--> integer list of rows, 2--> column with index 2
Z = my_dataset.iloc[[5, 10, 15], 2].values

# 2:25--> slice of rows, "Age"--> select column with label "Age"
W = my_dataset.loc[2:25, "Age"].values
