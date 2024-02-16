# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.linalg import svd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from math import sqrt
from fancyimpute import IterativeImputer


############################################################
# Load the Wine dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target
feature_names = wine_data.feature_names


############################################################
# Convert the dataset to a DataFrame
## Your code starts here
df = pd.DataFrame(X, columns=feature_names)
## Your code ends here

# Display the first few rows of the dataset
## Your code starts here
print(df.head())
## Your code ends here


############################################################
# make a copy of the original dataframe
df_original = df.copy()

# Introduce missing values in the dataset for demonstration purposes
# Replace some values with NaN to simulate missing data
df.iloc[10:15, 0] = np.nan
df.iloc[20:25, 1] = np.nan
df.iloc[30:35, 2] = np.nan

# Handling missing values using different methods and calculating RMSE
imputation_methods = ['mean', 'median', 'iterative']

# Write a code to print out the rmse metric for each strategy
## Your code starts here

for methods in imputation_methods:

    if methods == 'mean':
        column_means = df.mean()
        df_filled_means = df.fillna(column_means)
        print(f"Mean RMSE: {sqrt(mean_squared_error(df_original, df_filled_means))}")

    elif methods == 'median':
        column_medians = df.median()
        df_filled_medians = df.fillna(column_medians)
        print(f"Median RMSE: {sqrt(mean_squared_error(df_original, df_filled_medians))}")
    
    elif methods == 'iterative':
        iterative_imputer = IterativeImputer(random_state=0)
        iterative_imputed = iterative_imputer.fit_transform(df)
        df_filled_iterative = pd.DataFrame(iterative_imputed, columns=feature_names)
        print(f"Iterative RMSE: {sqrt(mean_squared_error(df_original, df_filled_iterative))}")
## Your code ends here
        



#################################################
# Exploratory Data Analysis
# Visualize key statistics
## Your code starts here
plt.figure(figsize=(12, 8))

sns.boxplot(data=df)

plt.title("Box Plot of Wine Dataset Features (Original)")

plt.show()
## Your code ends here


#######################
# Standardize the data
## Your code starts here
""" Put the standardized data back into dataframe"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.dropna())
df_scaled = pd.DataFrame(X_scaled, columns=feature_names)

## Your code ends here


#################################
# Exploratory Data Analysis
# Visualize key statistics
## Your code starts here
""" Use boxplot to show visualize the standardized features """
plt.figure(figsize=(25, 8))
sns.boxplot(data=df_scaled)
plt.title("Box Plot of Standardized Wine Dataset Features")
plt.show()

## Your code ends here


##########################
# Standardize the data
## Your code starts here
""" Put the standardized data back into dataframe"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.dropna())
df_scaled = pd.DataFrame(X_scaled, columns=feature_names)

## Your code ends here

#######################
# Calculate correlation matrix
## Your code starts here
correlation_matrix = df_scaled.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix)
plt.title("Correlation Matrix of Standardized Wine Dataset Features")
plt.show()

# Visualize correlation matrix

## Your code ends here



#################################
# Dimensionality Reduction using PCA
# Apply PCA
## Your code starts here
pca = PCA()
X_pca = pca.fit_transform(df_scaled)
## Your code ends here

# Determine the number of principal components to retain
## Your code starts here

## Your code ends here

# Plot explained variance ratio
## Your code starts here
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Components')
plt.grid(True)
plt.show()
## Your code ends here

# Choose the number of components based on the explained variance ratio
## Your code starts here
n_components = 10
## Your code ends here

# Perform PCA with the chosen number of components
## Your code starts here
pca_final = PCA(n_components=n_components)
X_pca_final = pca_final.fit_transform(df_scaled)
## Your code ends here

# Interpretation and Conclusion
# Interpret principal components
print("Explained variance ratio of each principal component:")
print(pca_final.explained_variance_ratio_)
## Your code starts here

## Your code ends here

# Summarize key findings
print("Summary:")
""" write your code inside the .format()!"""
total_variance = np.sum(pca_final.explained_variance_ratio_) * 100
print(f"PCA captures {total_variance:.2f}% of the variance with {n_components} components.")