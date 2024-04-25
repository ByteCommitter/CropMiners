# %%
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

# %%
df = pd.read_csv(r'D:\Acads - 2(2)\DM\Project\crop_recommendation.csv')
df.head()


# %%
missing_values=["N/A",np.nan,"na"]
df = pd.read_csv(r'crop_recommendation.csv',na_values=missing_values)

# %%
df.isnull().sum()

# %%
df.info()

# %%
plt.figure(figsize=(12,12))
i=1
for col in df.iloc[:,:-1]:
    plt.subplot(3,3,i)
    df[[col]].boxplot()
    i+=1

# %%
# Calculate Q1, Q3, and IQR for all columns except the last one
Q1 = df.iloc[:, :-1].quantile(0.25)
Q3 = df.iloc[:, :-1].quantile(0.75)
IQR = Q3 - Q1

# Define a mask for values that are NOT outliers
mask = ~((df.iloc[:, :-1] < (Q1 - 1.5 * IQR)) | (df.iloc[:, :-1] > (Q3 + 1.5 * IQR)))

# Apply the mask to df, keeping all rows in the last column
df_no_outliers = df[mask.all(axis=1)]
df_no_outliers

# %%
from sklearn.preprocessing import MinMaxScaler

# Separate the features from the labels
features = df_no_outliers.iloc[:, :-1]
labels = df_no_outliers.iloc[:, -1]

# Create the scaler
scaler = MinMaxScaler()

# Fit the scaler to the features and transform
scaled_features = scaler.fit_transform(features)

# Convert the scaled features back into a DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

# Add the labels back into the DataFrame
df_scaled['label'] = labels.values
df_scaled

# %%


# %%
from sklearn.ensemble import RandomForestClassifier

# Separate the features from the labels
features = df_scaled.iloc[:, :-1]
labels = df_scaled.iloc[:, -1]

# Initialize a random forest classifier
model = RandomForestClassifier()

# Fit the model to the data
model.fit(features, labels)

# Get the feature importances
importances = model.feature_importances_

# Get the feature names
feature_names = features.columns

# Combine the feature names and importances into a dictionary
feature_importances = dict(zip(feature_names, importances))

# Print the feature importances
for feature, importance in feature_importances.items():
    print(f"{feature}: {importance * 100 }% ")

# %%
from sklearn.decomposition import PCA

# Initialize a PCA
pca = PCA(n_components=2)

# Separate the features from the labels
features_scaled = df_scaled.iloc[:, :-1]
labels = df_scaled.iloc[:, -1]

# Perform PCA on the scaled features
pca_features = pca.fit_transform(features_scaled)

# Create a new DataFrame for the PCA features
df_pca_scaled = pd.DataFrame(data = pca_features, columns = ['principal component 1', 'principal component 2'])

# Add the labels to the DataFrame
df_pca_scaled['label'] = labels

# %%
import matplotlib.pyplot as plt

# Create a scatter plot of the two principal components
plt.figure(figsize=(8,6))
plt.scatter(df_pca_scaled['principal component 1'], df_pca_scaled['principal component 2'], c='blue')

# Set the title and labels of the plot
plt.title('Principal Component Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Show the plot
plt.show()

# %%

# Calculate the IQR of each column
Q1 = df_pca_scaled.iloc[:, :-1].quantile(0.25)
Q3 = df_pca_scaled.iloc[:, :-1].quantile(0.75)
IQR = Q3 - Q1

# Define the upper and lower bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove the outliers
df_pca_no_outliers = df_pca_scaled[~((df_pca_scaled.iloc[:, :-1] < lower_bound) | (df_pca_scaled.iloc[:, :-1] > upper_bound)).any(axis=1)]

print("Data after removing outliers:")
print(df_pca_no_outliers)

# %%
import matplotlib.pyplot as plt

# Create a scatter plot of the two principal components
plt.figure(figsize=(8,6))
plt.scatter(df_pca_no_outliers['principal component 1'], df_pca_no_outliers['principal component 2'], c='blue')

# Set the title and labels of the plot
plt.title('Principal Component Analysis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Show the plot
plt.show()

# %%



# %%


# %%
# Get the unique labels
unique_labels = df_scaled['label'].unique()

# Your existing code...
pca_dfs = {}

# Loop over the unique labels
for label in unique_labels:
    # Subset the DataFrame for the current label
    subset = df_scaled[df_scaled['label'] == label].iloc[:, :-1]
    
    # Perform PCA on the subset
    pca_features = pca.fit_transform(subset)
    
    # Create a new DataFrame for the PCA features
    df_pca = pd.DataFrame(data = pca_features, columns = ['PC1', 'PC2'])
    
    # Add the labels to the DataFrame
    df_pca['label'] = label
    
    # Store the DataFrame in the dictionary
    pca_dfs[label] = df_pca
    
    # Create a scatter plot of the PCA features
    plt.figure(figsize=(10, 7))
    plt.scatter(df_pca['PC1'], df_pca['PC2'])
    
    # Add labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'PCA of {label} Data')
    
    # Show the plot
    plt.show()

# %%
# Loop over the unique labels
for label in unique_labels:
    # Subset the DataFrame for the current label
    subset = df_scaled[df_scaled['label'] == label].iloc[:, :-1]
    
    # Perform PCA on the subset
    pca.fit(subset)
    
    # Get the feature importance for each Principal Component
    feature_importance = pd.DataFrame(pca.components_, columns=subset.columns, index=['PC1', 'PC2'])
    
    # Print the feature importance
    print(f"Feature importance for {label} data:")
    print(feature_importance)

# %%


# %%
from sklearn.model_selection import train_test_split

# Assume 'label' is the target variable and the rest are features
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
##SVM
from sklearn import svm
from sklearn.metrics import accuracy_score

# Define your SVM model
clf = svm.SVC()

# Train the model
clf.fit(X_train, y_train)

# Make predictions using the model
predictions = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# %%


# %%
#this is for GridSearchCV along with hypermetric testing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize a random forest classifier
rf = RandomForestClassifier()

# Initialize the grid search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train a new classifier using the best parameters
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train, y_train)


# Use the trained classifier to make predictions on the test set
predictions = best_rf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# Print the classification report
report = classification_report(y_test, predictions)
print(f"Classification Report: \n{report}")



# Perform cross-validation
scores = cross_val_score(best_rf, X_train, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average cross-validation score: {scores.mean()}")

# %%
from sklearn.metrics import classification_report
print(f"Model Accuracy: {accuracy}")
# Print the classification report
report = classification_report(y_test, predictions)
print(f"Classification Report: \n{report}")
# Print the confusion matrix

# %%
from sklearn.model_selection import train_test_split

# 'label' is the target variable and the rest are features
X = df.drop('label', axis=1)
y = df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Define your GBM model
gbm = GradientBoostingClassifier()

# Define your parameter grid for GBM
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 1],
    # Add any other parameters you want to tune
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best model
best_gbm = grid_search.best_estimator_

# Make predictions using the best model
predictions = best_gbm.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# %%
from sklearn.metrics import classification_report

# Make predictions using the best GBM model
gbm_predictions = best_gbm.predict(X_test)

# Generate a classification report for GBM
gbm_report = classification_report(y_test, gbm_predictions)

print("GBM Classification Report: ")
print(gbm_report)

# Define your specific conditions and resources
input_data = {
    'N': [31],
    'P': [70],
    'K': [77],
    'temperature': [20],
    'humidity': [14],
    'ph': [6],
    'rainfall': [80],
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame(input_data)

# Use the best GBM model to predict the crop
predicted_crop = best_gbm.predict(input_df)

print(f"The recommended crop to grow according to GBM is: {predicted_crop[0]}")

# Make predictions using the SVM model
svm_predictions = clf.predict(X_test)

# Generate a classification report for SVM
svm_report = classification_report(y_test, svm_predictions)

print("SVM Classification Report: ")
print(svm_report)


# Use the SVM model to predict the crop
predicted_crop_svm = clf.predict(input_df)
print(f"The recommended crop to grow according to SVM is: {predicted_crop_svm[0]}")

# Use the RandomForest model to predict the crop
predicted_crop_rf = best_rf.predict(input_df)
print(f"The recommended crop to grow according to RandomForest is: {predicted_crop_rf[0]}")

# %%
from sklearn import svm
from sklearn.metrics import accuracy_score

# Define your SVM model
clf = svm.SVC()

# Train the model
clf.fit(X_train, y_train)

# Make predictions using the model
predictions = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy}")

# %%
##Random forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Initialize the Random Forest Classifier
rf = RandomForestClassifier()

# Fit the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
rf_predictions = rf.predict(X_test)

# Generate a classification report
rf_report = classification_report(y_test, rf_predictions)

print("Random Forest Classification Report: ")
print(rf_report)

# %%



