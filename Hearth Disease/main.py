import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("framingham.csv")

# Drop rows with missing values
data = data.dropna()
print(data.shape)

# Plot 1: Gender, 10-year CHD risk, and diabetes
sns.barplot(x="male", y="TenYearCHD", hue="diabetes", data=data)
plt.show()

# Plot 2: Gender, 10-year CHD risk, and smoking status
sns.barplot(x="male", y="TenYearCHD", hue="currentSmoker", data=data)
plt.show()

# Convert TenYearCHD column to 0 and 1
data["TenYearCHD"] = data["TenYearCHD"].apply(lambda x: 1 if x == 1 else 0)

# Define target variable (y) and features (x_data)
y = data["TenYearCHD"].values
x_data = data.drop(["TenYearCHD"], axis=1)

# Normalize the feature data
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# No need to reshape x_train and x_test; they should remain 2D
# Only y_train and y_test need to be 1D
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# Fit the model
lr.fit(x_train, y_train)

# Test accuracy
print("Test accuracy: {}".format(lr.score(x_test, y_test)))



