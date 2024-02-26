# Import the neccesary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#################################################
# Set random seed for reproducibility
np.random.seed(42)

# Generate instances for class one using normal distribution
class_one_mean = (10, 10)  # Center around (10, 10)
class_one_std = 1
class_one_instances = 1000
class_one = np.random.normal(loc=class_one_mean, scale=class_one_std, size=(class_one_instances, 2))

# Generate noisy instances for class one
noisy_instances = 400
class_one_noisy = np.random.normal(loc=(0, 0), scale=3, size=(noisy_instances, 2))

# Generate instances for class two using uniform distribution in a [0,20] range]
class_two_instances = 1400
class_two = np.random.uniform(low=0, high=20, size=(class_two_instances, 2))

# Concatenate class one instances and noisy instances
X_class_one = np.vstack((class_one, class_one_noisy + class_one_mean))  # Adjusting mean to add noise
y_class_one = np.zeros(X_class_one.shape[0])  # Label for class one is 0

# Labels for class two are 1
X_class_two = class_two
y_class_two = np.ones(class_two.shape[0])

# Concatenate features and labels for both classes
X = np.vstack((X_class_one, X_class_two))
y = np.concatenate((y_class_one, y_class_two))


# Visualize the data
plt.figure(figsize=(10, 6))

# Plot instances for class one
plt.scatter(X_class_one[:, 0], X_class_one[:, 1], color='blue', label='Class One (Normal)', alpha=0.5)

# Plot instances for class two
plt.scatter(X_class_two[:, 0], X_class_two[:, 1], color='green', label='Class Two (Uniform)', alpha=0.5)

plt.title('Generated Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()


##############################
# Split the dataset into training and testing sets with 20% test and 80% training
#Your code starts here
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Your code ends here

# Train a decision tree classifier with default parameters
#You code starts here
dtClassifier = DecisionTreeClassifier(random_state=0)
dtClassifier.fit(x_train, y_train)
#Your code ends here

# Evaluate the classifier on the training set
#You code starts here
y_trainPrediction = dtClassifier.predict(x_train)
trainAccuracy = accuracy_score(y_train, y_trainPrediction)
#Your code ends here

# Evaluate the classifier on the testing set
#You code starts here
y_testPrediction = dtClassifier.predict(x_test)
testAccuracy = accuracy_score(y_test, y_testPrediction)
#Your code ends here


# Visualize the learned tree
""" use plot_tree from sklearn.tree """
#You code starts here

plt.figure(figsize=(20, 10))  # Set the figure size for better visibility
plot_tree(dtClassifier, 
          filled=True, 
          rounded=True, 
          class_names=["Class One", "Class Two"], 
          feature_names=["Feature 1", "Feature 2"],
          #max_depth=3
          ) 
plt.title("Decision Tree Visualization")
plt.show()

#Your code ends here

#####################################
# Define the range of max_depth values to test
max_depth_values = range(1, 21)  # Test max_depth from 1 to 20

# Initialize lists to store training and test errors
training_errors = []
test_errors = []

# Split the dataset into training and testing sets with 20% test and 80% training
#You code starts here
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Your code ends here

# Iterate over each max_depth value
#You code starts here
for max_depth in max_depth_values:
    # Train a decision tree classifier with the current max_depth value
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(x_train, y_train)

    # Evaluate the classifier on the training set
    y_train_pred = clf.predict(x_train)
    training_error = 1 - accuracy_score(y_train, y_train_pred)
    training_errors.append(training_error)

    # Evaluate the classifier on the test set
    y_test_pred = clf.predict(x_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)
#Your code ends here

# Plot the sensitivity analysis results

plt.figure(figsize=(10, 6))
#You code starts here
plt.plot(max_depth_values, training_errors, label='Training Error', marker='o')
plt.plot(max_depth_values, test_errors, label='Test Error', marker='s')

#Your code ends here
plt.title('Sensitivity Analysis over Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Classification Error')
plt.xticks(max_depth_values)
plt.grid(True)
plt.legend()
plt.show()


#################################
# Split the dataset into training and testing sets with 20% test and 80% training
#You code starts here
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Your code ends here

# Train a decision tree classifier with a maximum depth of 20
#You code starts here
clf = DecisionTreeClassifier(max_depth=20, random_state=0)
clf.fit(x_train, y_train)
#Your code ends here

# Evaluate the classifier on the training set
#You code starts here
y_train_pred = clf.predict(x_train)
training_error = 1 - accuracy_score(y_train, y_train_pred)
training_errors.append(training_error)
#Your code ends here

# Evaluate the classifier on the testing set
#You code starts here
y_test_pred = clf.predict(x_test)
test_error = 1 - accuracy_score(y_test, y_test_pred)
test_errors.append(test_error)
#Your code ends here


#####################################
# Split the dataset into training and testing sets with 20% test and 80% training
#You code starts here
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#You code ends here

# Train a decision tree classifier with a maximum depth of 4
#You code starts here
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf.fit(x_train, y_train)
#You code ends here

# Evaluate the classifier on the training set
#You code starts here
y_train_pred = clf.predict(x_train)
training_error = 1 - accuracy_score(y_train, y_train_pred)
training_errors.append(training_error)
#You code ends here

# Evaluate the classifier on the testing set
#You code starts here
y_test_pred = clf.predict(x_test)
test_error = 1 - accuracy_score(y_test, y_test_pred)
test_errors.append(test_error)



#########################################
# Split the dataset into training and testing sets with 20% test and 80% training
#You code starts here
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#You code ends here

# Train a decision tree classifier with a maximum size of 20
""" You can use the same code as before. You have already trained this model."""
#You code starts here
tree_clf = DecisionTreeClassifier(max_depth=20, random_state=0)
tree_clf.fit(x_train, y_train)
#You code ends here

# Apply post-pruning with different ccp_alpha values
""" use tree_clf.cost_complexity_pruning_path , then capture ccp_alphas """
#You code starts here
path = tree_clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
#You code ends here

# Initialize lists to store accuracy scores
accuracy_scores = []

# Iterate over different ccp_alpha values and retrain the decision tree and print out the accuracy of the retrained model
#You code starts here
for ccp_alpha in ccp_alphas:

    tree_clf_pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    tree_clf_pruned.fit(x_train, y_train)

    y_pred_pruned = tree_clf_pruned.predict(x_test)

    accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
    accuracy_scores.append(accuracy_pruned)

    print(f"Accuracy for ccp_alpha={ccp_alpha}: {accuracy_pruned}")
#You code ends here