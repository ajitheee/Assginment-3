# decision_tree.py
# Complete solution with preprocessing fix and sklearn compatibility
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Training datasets and test set
dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']
test_set = 'cheat_test.csv'

# Repeat the training and testing process
for ds in dataSets:
    accuracies = []

    for _ in range(10):
        # Load training data
        df_train = pd.read_csv(ds)

        # Preprocess 'Taxable Income' column (remove 'k' and convert to float)
        df_train['Taxable Income'] = df_train['Taxable Income'].replace({'k':'000'}, regex=True).astype(float)

        # Feature processing (One-hot encoding Marital Status)
        X_train_categorical = df_train[['Refund', 'Marital Status']]
        X_train_numeric = df_train[['Taxable Income']]

        enc = OneHotEncoder(drop='first', sparse_output=False)  # Compatibility fix
        X_train_encoded = enc.fit_transform(X_train_categorical)

        X_train = np.hstack((X_train_encoded, X_train_numeric))

        # Convert class labels to binary
        Y_train = df_train['Cheat'].map({'Yes': 1, 'No': 0})

        # Train decision tree
        clf = DecisionTreeClassifier(criterion='gini')
        clf.fit(X_train, Y_train)

        # Load and preprocess test data
        df_test = pd.read_csv(test_set)
        df_test['Taxable Income'] = df_test['Taxable Income'].replace({'k':'000'}, regex=True).astype(float)

        X_test_categorical = df_test[['Refund', 'Marital Status']]
        X_test_numeric = df_test[['Taxable Income']]

        X_test_encoded = enc.transform(X_test_categorical)
        X_test = np.hstack((X_test_encoded, X_test_numeric))

        Y_test = df_test['Cheat'].map({'Yes': 1, 'No': 0})

        # Predict and calculate accuracy
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
        accuracies.append(accuracy)

    avg_accuracy = np.mean(accuracies)
    print(f'Final accuracy when training on {ds}: {avg_accuracy:.2f}')