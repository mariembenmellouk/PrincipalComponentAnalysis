import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm  
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats



# Load dataset 
data = pd.read_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task3-D600\D600 Task 3 Dataset 1 Housing Information.csv")

# Select the continuous variables to standardize
continuous_vars = ['NumBathrooms', 'NumBedrooms', 'Price']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply standardization
data[continuous_vars] = scaler.fit_transform(data[continuous_vars])

# Save the cleaned dataset to a new CSV
data.to_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task3-D600\Cleaned Dataset.csv", index=False)

# Display the cleaned dataset
print(data.head())

# Calculate descriptive statistics for these variables
descriptive_stats = data[continuous_vars].describe()

# Calculate the mode for each variable
modes = data[continuous_vars].mode().iloc[0]

# Print out descriptive statistics
print("Descriptive Statistics:")
print(descriptive_stats)

# Print out modes
print("\nMode values:")
print(modes)

# Apply PCA
pca = PCA()

# Perform PCA and get the transformed data (principal components)
pca_components = pca.fit_transform(data[continuous_vars])

# Get the matrix of principal components
components = pca.components_

# Print the principal components matrix
print("Principal Components Matrix:")
print(components)

# Get the eigenvalues (explained variance)
eigenvalues = pca.explained_variance_

# Print the eigenvalues
print("Eigenvalues of the principal components:", eigenvalues)

# Apply the Kaiser Rule: Retain components with eigenvalue > 1
components_retained = np.sum(eigenvalues > 1)

# Print the number of components to retain
print(f"Number of components to retain according to the Kaiser Rule: {components_retained}")

# Calculate the variance of each component)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio of the principal components:", explained_variance_ratio)

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
print("Cumulative explained variance:", cumulative_variance)

# Split the data into training and test sets using the retained components from PCA
X_train, X_test, y_train, y_test = train_test_split(pca_components[:, :components_retained], data['Price'], test_size=0.2, random_state=42)

# Combine the features and target variable into one DataFrame for training and test datasets
train_pca_data = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])])  
train_pca_data['Price'] = y_train  
test_pca_data = pd.DataFrame(X_test, columns=[f'PC{i+1}' for i in range(X_test.shape[1])])  
test_pca_data['Price'] = y_test  

# Save the training and test datasets into CSV files
train_pca_data.to_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task3-D600\train_pca_data.csv", index=False)
test_pca_data.to_csv(r"C:\Users\merie\OneDrive\Bureau\WGU\D600\Task3-D600\test_pca_data.csv", index=False)

# Print confirmation
print("PCA Datasets saved: train_pca_data.csv, test_pca_data.csv")

# Assumption Checks
# Define independent variables and a constant 
X = data[['NumBathrooms', 'NumBedrooms']]
X = sm.add_constant(X)
# Define the dependent variable
y = data['Price']
# Fit the model
model = sm.OLS(y, X).fit()
# residual and scatter plots to verify linearity
plt.figure(figsize=(8, 6))
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

#Normality of Residuals
plt.figure(figsize=(8, 6))
stats.probplot(model.resid, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
shapiro_test = stats.shapiro(model.resid)
print(f'Shapiro-Wilk test statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}')

#Constatnt variance (Breusch-Pagan test)
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
print(dict(zip(labels, bp_test)))

# Summary of the model
print(model.summary())

# Add constant (intercept) to the features
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Backward Elimination Process
def backward_elimination(X, y, significance_level=0.05):
    while True:
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Print the p-values of the model
        print(f"Current model p-values:\n{model.pvalues}\n")
        
        # Find the maximum p-value
        max_p_value = model.pvalues.max()
        
        # If the maximum p-value is greater than the significance level, drop the feature
        if max_p_value > significance_level:
            excluded_variable = model.pvalues.idxmax()  
            # Get the column with the max p-value
            print(f"Dropping variable: {excluded_variable} with p-value: {max_p_value}")
            
            # Find the column index to drop based on the excluded variable
            excluded_index = model.pvalues.index.get_loc(excluded_variable)

            # Exclude the variable from x
            X = np.delete(X, excluded_index, axis=1)
         
        else:
            break  # All features are significant, stop 
    
    return model

# Perform backward elimination on the training set
optimized_model_pca = backward_elimination(X_train, y_train)

# Print summary of the optimized model
print("Optimized Model Summary with PCA:")
print(optimized_model_pca.summary())

# Extract model parameters
params = {
    "Adjusted R²": optimized_model_pca.rsquared_adj,
    "R²": optimized_model_pca.rsquared,
    "F-statistic": optimized_model_pca.fvalue,
    "Probability F-statistic": optimized_model_pca.f_pvalue,
    "Coefficient Estimates": optimized_model_pca.params,
    "P-values": optimized_model_pca.pvalues
}

# Print model parameters
for key, value in params.items():
    print(f"{key}: {value}")

# Prepare the predictors for the optimized model
optimized_features = optimized_model_pca.model.exog

# Make predictions on the training set using the optimized model
predictions_train = optimized_model_pca.predict(optimized_features)

# Calculate Mean Squared Error (MSE) on the training set
mse_train = np.mean((y_train - predictions_train) ** 2)
print(f"\nMean Squared Error of the optimized model (training set): {mse_train:.2f}")

# Extract model parameters for the regression equation
intercept = optimized_model_pca.params['const'] if 'const' in optimized_model_pca.params else 0
coefficients = optimized_model_pca.params.drop('const', errors='ignore')

# Print the regression equation
regression_equation = f"Price = {intercept:.2f}"
for i, coef in coefficients.items():
    regression_equation += f" + {coef:.2f} * {i}"

print("\nRegression Equation:")
print(regression_equation)

# Discuss the coefficients
print("\nDiscuss the coefficients:")
print(f"Intercept : {intercept:.2f} - This is the expected price when all the principal components are zero.")
for feature, coef in coefficients.items():
    print(f"{feature} : {coef:.2f} - This indicates that for each unit increase in {feature}, the price is expected to increase by {coef:.2f}")

# Use only the first principal component 
X_test_pca = X_test[:, :1]
# Make predictions on the test set using the optimized model
test_predictions = optimized_model_pca.predict(X_test_pca)

# Calculate the mean squared error for the test set
mse_test = np.mean((y_test - test_predictions) ** 2)

# Print the mean squared errors for training and test sets
print(f"The MSE for the training set is {mse_train:.2f}, while the MSE for the test set is {mse_test:.2f}.")

# Compare the MSE values to evaluate model performance
if mse_train < mse_test:
    print("The training set MSE is lower than the test set MSE, suggesting that the model may be overfitting the training data.")
elif mse_train > mse_test:
    print("The training set MSE is higher than the test set MSE, suggesting that the model may be underfitting the training data.")
else:
    print("The training set MSE is similar to the test set MSE, indicating that the model is generalizing well.")