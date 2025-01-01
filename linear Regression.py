# Import necessary libraries
import numpy as np                # For numerical operations
import pandas as pd               # For data manipulation and analysis
import matplotlib.pyplot as plt    # For data visualization
from sklearn import linear_model   # For linear regression model
from sklearn.metrics import r2_score # For performance evaluation metrics

# Load the dataset
df = pd.read_csv("house.csv")     # Read the CSV file containing house data

# Split the data into training and testing sets
# 80% of the data will be used for training, and 20% for testing
msk = np.random.rand(len(df)) < 0.8  # Create a boolean mask for splitting
train = df[msk]                       # Training data
test = df[~msk]                       # Testing data

# Create a linear regression model
regr = linear_model.LinearRegression()

# Create a figure for plotting
fig = plt.figure()
pic1 = fig.add_subplot(111)          # Add a subplot to the figure

# Set the labels for the axes
plt.xlabel("Area")                   # X-axis label
plt.ylabel("Price")                  # Y-axis label

# Prepare the training and testing data
x_train = np.asanyarray(train["Area"])  # Features for training
y_train = np.asanyarray(train["Price"]) # Target variable for training
x_test = np.asanyarray(test["Area"])    # Features for testing
y_test = np.asanyarray(test["Price"])   # Target variable for testing

# Train the model using the training data
regr.fit(x_train.reshape((-1, 1)), y_train)  # Fit the model

# Make predictions using the testing data
y_test_ = regr.predict(x_test.reshape((-1, 1)))  # Predicted prices

# Print evaluation metrics
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(y_test - y_test_)))
print("Residual Sum of Squares (MSE): %.2f" % np.mean((y_test - y_test_) ** 2))
print("R^2 Score: %.2f" % r2_score(y_test, y_test_))

# Plotting the results
pic1.scatter(x_train, y_train, color="red", label='Training data')       # Plot training data
pic1.scatter(x_test, y_test, color="blue", label='Testing data')         # Plot testing data
pic1.scatter(x_test, y_test_, color="yellow", label='Predicted data')    # Plot predicted data
pic1.plot(x_train, x_train * regr.coef_ + regr.intercept_, color="green", label='Regression line') # Plot regression line

# Show the legend and the plot
plt.legend()
plt.show()
