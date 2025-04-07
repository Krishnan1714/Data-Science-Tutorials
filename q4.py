import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris flower dataset
# This dataset has 150 samples from 3 species: Setosa, Versicolor, Virginica
# Each sample has 4 features: sepal length, sepal width, petal length, petal width
# The task is to classify the flower species based on the given features
iris_dataset = load_iris()
features, labels = iris_dataset.data, iris_dataset.target

# Split the dataset into training and testing parts
# We'll use 70% of the data for training, and the remaining 30% for testing
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

# Configuration for our custom random forest
total_trees = 10           # Total number of decision trees in our forest
seed_value = 42            # Random seed for reproducibility

# Train multiple decision trees
# Each tree is trained with a different random state for variation
forest_models = []
for tree_index in range(total_trees):
    tree_model = DecisionTreeClassifier(random_state=seed_value + tree_index)
    tree_model.fit(features_train, labels_train)
    forest_models.append(tree_model)

# Collect predictions from each decision tree
# Each model predicts on the test set, and results are stored
all_predictions = []
for model in forest_models:
    all_predictions.append(model.predict(features_test))
all_predictions = np.array(all_predictions)

# Average the predictions across all trees (like soft voting)
# Then round to the nearest integer to determine the final class
mean_predictions = np.mean(all_predictions, axis=0)
ensemble_predictions = np.round(mean_predictions).astype(int)

# Evaluate how accurate the final prediction is
# This compares our ensemble output with the true test labels
ensemble_accuracy = accuracy_score(labels_test, ensemble_predictions)
print(f"Accuracy of Averaged Decision Tree Ensemble: {ensemble_accuracy * 100:.2f}%")
