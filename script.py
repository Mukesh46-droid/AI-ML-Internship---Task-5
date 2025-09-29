import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. Load Dataset ---
# Load the Heart Disease dataset from a reliable online source (UCI repository)
url = 'http://storage.googleapis.com/download.tensorflow.org/data/heart.csv'
df = pd.read_csv(url)

# Prepare the data
# The 'thal' column has a value '0' which is not documented, let's remove it for cleaner data
df = df[df['thal'] != '0']
# Convert categorical 'thal' column to numeric
df['thal'] = pd.to_numeric(df['thal'])

# Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Dataset successfully loaded and prepared ---")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}\n")


# --- 2. Train and Analyze a Decision Tree ---
# Train a full-depth Decision Tree to demonstrate potential overfitting
full_depth_tree = DecisionTreeClassifier(random_state=42)
full_depth_tree.fit(X_train, y_train)
print(f"Full-depth Decision Tree accuracy: {accuracy_score(y_test, full_depth_tree.predict(X_test)):.4f}")

# Train a pruned Decision Tree to prevent overfitting by controlling its depth
pruned_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
pruned_tree.fit(X_train, y_train)
pruned_tree_accuracy = accuracy_score(y_test, pruned_tree.predict(X_test))
print(f"Pruned Decision Tree (max_depth=4) accuracy: {pruned_tree_accuracy:.4f}\n")

# Visualize the pruned decision tree
print("--- Generating visualization for the pruned Decision Tree... ---")
plt.figure(figsize=(20, 10))
plot_tree(pruned_tree,
          feature_names=X.columns,
          class_names=['No Disease', 'Disease'],
          filled=True,
          rounded=True)
plt.title("Pruned Decision Tree (max_depth=4)")
plt.savefig("decision_tree_visualization.png") # Saves the plot to a file
print("Visualization saved as 'decision_tree_visualization.png'\n")


# --- 3. Train a Random Forest and Compare Accuracy ---
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
print("--- Random Forest Model ---")
print(f"Random Forest accuracy: {rf_accuracy:.4f}")

print("\n--- Accuracy Comparison ---")
print(f"Pruned Decision Tree: {pruned_tree_accuracy:.4f}")
print(f"Random Forest:        {rf_accuracy:.4f}\n")


# --- 4. Interpret Feature Importances from Random Forest ---
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns).sort_values(ascending=False)

print("--- Top 5 Feature Importances from Random Forest ---")
print(feature_importances.head(), "\n")

# Plot feature importances
plt.figure(figsize=(12, 7))
feature_importances.plot(kind='bar')
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("Feature importance plot saved as 'feature_importances.png'\n")


# --- 5. Evaluate Models Using Cross-Validation ---
# Use 5-fold cross-validation to get a more robust measure of model performance
cv_scores_tree = cross_val_score(pruned_tree, X, y, cv=5)
cv_scores_rf = cross_val_score(rf_classifier, X, y, cv=5)

print("--- Model Evaluation with 5-Fold Cross-Validation ---")
print(f"Pruned Decision Tree CV Mean Accuracy: {np.mean(cv_scores_tree):.4f} (+/- {np.std(cv_scores_tree):.4f})")
print(f"Random Forest CV Mean Accuracy:        {np.mean(cv_scores_rf):.4f} (+/- {np.std(cv_scores_rf):.4f})")
