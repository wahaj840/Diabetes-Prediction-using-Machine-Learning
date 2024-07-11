import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('notebook/data/diabetes.csv')

# Split the data into features (X) and target variable (y)
X = df[[ 'Glucose', 'BMI','DiabetesPedigreeFunction' ,'Age']]
y = df['Outcome']

# To Calculate mean and standard deviation for each feature
means = X.mean()
std_devs = X.std()

# Define the number of standard deviations for outlier detection
num_std_devs = 2

# Calculate lower and upper bounds for outlier detection
lower_bounds = means - (num_std_devs * std_devs)
upper_bounds = means + (num_std_devs * std_devs)

# Filter out input values that fall outside the lower and upper bounds
outliers = ((X < lower_bounds) | (X > upper_bounds)).any(axis=1)
X = X[~outliers]
y = y[~outliers]

# print(X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

oversample = RandomOverSampler(sampling_strategy='minority')

X_over, y_over = oversample.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_over, y_over)

# Save the testing data to files
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Make predictions on the test set
y_pred = model.predict(X_test)
# print(y_pred)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)

pickle.dump(model, open('logistic_reg_model.pkl', 'wb'))