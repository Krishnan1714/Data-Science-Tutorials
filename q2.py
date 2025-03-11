import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define dataset
sales_data = {
    "Ad_Budget": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "Success_Flag": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 1 = Successful Sale, 0 = No Sale
}

# Convert dictionary to DataFrame
data_frame = pd.DataFrame(sales_data)

# Split into input features and target variable
input_feature = data_frame[["Ad_Budget"]]
output_label = data_frame["Success_Flag"]

# Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_feature, output_label, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict sale outcome for a given advertising budget
test_input = [[95]]  # Ad budget = $95
predicted_result = logistic_model.predict(test_input)

# Display prediction result
print(f"Predicted Sale Outcome for Ad Budget {test_input[0][0]}: {'Success' if predicted_result[0] == 1 else 'Failure'}")
