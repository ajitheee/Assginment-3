# -------------------------------------------------------------------------
# AUTHOR: Ajith Elumalai
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 1 day
# -----------------------------------------------------------*/
# roc_curve.py

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# Read dataset
df = pd.read_csv('cheat_data.csv')

# Preprocess 'Taxable Income': remove 'k' and convert to float
df['Taxable Income'] = df['Taxable Income'].replace({'k': '000'}, regex=True).astype(float)
# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
# Manually one-hot encode 'Marital Status'
df['Single'] = (df['Marital Status'] == 'Single').astype(int)
df['Divorced'] = (df['Marital Status'] == 'Divorced').astype(int)
df['Married'] = (df['Marital Status'] == 'Married').astype(int)

# Encode 'Refund': Yes = 1, No = 0
df['Refund'] = (df['Refund'] == 'Yes').astype(int)

# Construct feature matrix X and target vector y
X = df[['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income']].values
# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
# --> add your Python code here
y = df['Cheat'].map({'Yes': 1, 'No': 0}).values

# Split into train/test sets using 30% for testing
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)

# Generate no-skill prediction probabilities (all 0s for class 1)
ns_probs = [0 for _ in range(len(testy))]

# Train decision tree classifier
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# Predict probabilities for class 1
dt_probs = clf.predict_proba(testX)[:, 1]

# Calculate ROC AUC scores
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# Print AUC scores
print('No Skill: ROC AUC=%.3f' % ns_auc)
print('Decision Tree: ROC AUC=%.3f' % dt_auc)

# Calculate ROC curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# Plot ROC curves
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.title('ROC Curve - Decision Tree vs No Skill')
pyplot.grid(True)
pyplot.show()