import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Import the data
data_path = 'mushrooms.csv'
data = pd.read_csv(data_path)

#%% Preprocessing of data
s = data.dtypes == 'object'
catergorical_data = list(data[s[s].index])
data = data.apply(LabelEncoder().fit_transform)

# Split the data into features and output, and test and train
Y = data['class']
X = data[data.columns[1:]]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, random_state = 0)

#%% Testing the model for varying depths for a large number of leaf nodes
ers = [] # errors
cvMeans = [] # Cross-Valdiation means
cvStds = [] # CV standard deviations
# Loop through with varying tree depth
for depth in range(1, 11):
    model = DecisionTreeRegressor(max_depth=depth, random_state = 1)
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    ers.append(mean_absolute_error(Ytest, Ypred))
    cv = cross_val_score(model, Xtrain, Ytrain, cv=5)
    cvMeans.append(np.mean(cv))
    cvStds.append(np.std(cv))
# Plot the error
plt.figure()
plt.plot(range(1, 11), ers, marker='o', linestyle='--')
plt.xlabel("Depth")
plt.ylabel("Error")
plt.yscale("log")

# Plot CV data
plt.figure()
plt.errorbar(range(1, 11), cvMeans, yerr=cvStds, marker='o', linestyle='--')
plt.xlabel("Depth")
plt.ylabel("Cross-Validation Mean")

#%% Testing the model for varying the number of nodes
maxNodes = 25
nodes = range(2, maxNodes+1)
ers = [] # errors
cvMeans = [] # Cross-Valdiation means
cvStds = [] # CV standard deviations
# Loop through with varying tree depth
for node in nodes:
    model = DecisionTreeRegressor(max_leaf_nodes=node, random_state = 1)
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    ers.append(mean_absolute_error(Ytest, Ypred))
    cv = cross_val_score(model, Xtrain, Ytrain, cv=5)
    cvMeans.append(np.mean(cv))
    cvStds.append(np.std(cv))
# Plot the error
plt.figure()
plt.plot(nodes, ers, marker='o', linestyle='--', color='Red')
plt.xlabel("Maximum Leaf Nodes")
plt.ylabel("Error")
plt.yscale("log")

# Plot CV data
plt.figure()
plt.errorbar(nodes, cvMeans, yerr=cvStds, marker='o', linestyle='--', color='Red')
plt.xlabel("Maximum Leaf Nodes")
plt.ylabel("Cross-Validation Mean")

#%% Using less features
fts = ['cap-color', 'bruises', 'gill-color']
Xtrain = Xtrain[fts]
Xtest = Xtest[fts]
ers = [] # errors
cvMeans = [] # Cross-Valdiation means
cvStds = [] # CV standard deviations
# Loop through with varying tree depth
for depth in range(1, 51):
    model = DecisionTreeRegressor(max_depth=depth, random_state = 1)
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    ers.append(mean_absolute_error(Ytest, Ypred))
    cv = cross_val_score(model, Xtrain, Ytrain, cv=5)
    cvMeans.append(np.mean(cv))
    cvStds.append(np.std(cv))
# Plot the error
plt.figure()
plt.plot(range(1, 51), ers, marker='o', linestyle='--')
plt.xlabel("Depth")
plt.ylabel("Error")
plt.yscale("log")

# Plot CV data
plt.figure()
plt.errorbar(range(1, 51), cvMeans, yerr=cvStds, marker='o', linestyle='--')
plt.xlabel("Depth")
plt.ylabel("Cross-Validation Mean")