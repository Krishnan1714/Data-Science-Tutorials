import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Load sample dataset
data_frame = pd.read_csv("advertising.csv")

# Define independent (input) and dependent (output) variables
features = data_frame[['TV', 'radio', 'newspaper']]  # Input variables
target = data_frame['sales']  # Output variable
features = sm.add_constant(features)  # Add constant term for regression

# Train the Multiple Linear Regression model
regression_model = sm.OLS(target, features).fit()
r_squared_value = regression_model.rsquared
residual_sum_squares = sum((regression_model.resid) ** 2)  # Calculate RSS
residual_std_error = np.sqrt(residual_sum_squares / (len(target) - len(features.columns)))  # Compute RSE

# Compute F-statistic and corresponding p-value
f_stat = regression_model.fvalue
f_p_val = regression_model.f_pvalue

# Display results
print(f"R² Score: {r_squared_value:.4f}")
print(f"Residual Standard Error: {residual_std_error:.4f}")
print(f"F-Statistic: {f_stat:.4f}")
print(f"P-Value (F-Test): {f_p_val:.4e}")

# Show model summary
print(regression_model.summary())
