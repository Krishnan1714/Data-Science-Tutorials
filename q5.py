import numpy as np

# Prepare training data
features = np.array([[2, 3], [3, 4], [1, 2], [5, 6], [6, 5], [7, 8]]) 
labels = np.array([-1, -1, -1, 1, 1, 1])  # Binary labels

# Initialize parameters
weights = np.zeros(features.shape[1])
bias = 0
learning_rate = 0.01
regularization_strength = 0.01
num_epochs = 1000

# Train the model using gradient descent
for epoch in range(num_epochs):
    for i in range(len(features)):
        condition = labels[i] * (np.dot(features[i], weights) + bias) >= 1
        if condition:
            weights -= learning_rate * (2 * regularization_strength * weights)
        else:
            weights -= learning_rate * (2 * regularization_strength * weights - labels[i] * features[i])
            bias -= learning_rate * labels[i]

# Test the model
test_data = np.array([[4, 5], [2, 2], [6, 6]])
predicted_labels = np.sign(np.dot(test_data, weights) + bias)

print("Trained Weight Vector:", weights)
print("Trained Bias Term:", bias)
print("Predictions for Test Samples:", predicted_labels)
