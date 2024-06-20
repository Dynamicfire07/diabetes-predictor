import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical columns
le = LabelEncoder()
dataset['smoking_history'] = le.fit_transform(dataset['smoking_history'])
dataset['gender'] = le.fit_transform(dataset['gender'])

# Split the dataset into features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the logistic regression model
logreg = LogisticRegression(random_state=0,max_iter=500)
logreg.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_logreg)
print(f'Accuracy: {accuracy:.2f}')

# Display prediction results
results = np.concatenate((y_pred_logreg.reshape(len(y_pred_logreg), 1), y_test.reshape(len(y_test), 1)), axis=1)
print(results)

# Make a prediction for a new sample
new_sample = [[1, 45, 1, 0, 1, 26.8, 6.8, 180]]
print(logreg.predict(new_sample))
