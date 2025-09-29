# AI & ML Internship - Task 5: Decision Trees and Random Forests

This repository contains the solution for Task 5 of the Elevate Labs AI & ML Internship. The objective of this task is to understand, implement, and evaluate tree-based models: Decision Trees and Random Forests.

## Project Overview

This project involves the following steps:
1.  **Data Loading and Preparation**: The Heart Disease dataset from the UCI Machine Learning Repository is loaded and preprocessed for modeling.
2.  **Decision Tree Implementation**: A Decision Tree Classifier is trained. To demonstrate the concept of overfitting, an initial unconstrained tree is built, followed by a pruned tree with a controlled `max_depth` of 4.
3.  **Model Visualization**: The pruned decision tree is visualized to understand its decision-making logic.
4.  **Random Forest Implementation**: A Random Forest Classifier is trained and its accuracy is compared against the pruned decision tree.
5.  **Feature Importance Analysis**: The feature importances from the Random Forest model are extracted and visualized to identify the most influential factors in predicting heart disease.
6.  **Cross-Validation**: Both the pruned Decision Tree and the Random Forest models are evaluated using 5-fold cross-validation to assess their performance more robustly.

## Files in this Repository

*   `main.py`: The Python script containing all the code for loading data, training models, and performing evaluations.
*   `decision_tree_visualization.png`: An image file showing the structure of the pruned Decision Tree.
*   `feature_importances.png`: A bar chart illustrating the importance of each feature in the Random Forest model.
*   `README.md`: This file.

## Results and Conclusion

*   **Accuracy Comparison**:
    *   Pruned Decision Tree Accuracy: **90.16%**
    *   Random Forest Accuracy: **85.25%**
*   **Cross-Validation Accuracy**:
    *   Pruned Decision Tree (5-fold CV): **76.53%**
    *   Random Forest (5-fold CV): **82.83%**

Interestingly, on the single test split, the simpler pruned tree performed better. However, the cross-validation results, which are more reliable, show that the **Random Forest is the more robust and higher-performing model** on average across different data subsets.

*   **Key Feature Importances**: The most important features for predicting heart disease according to the Random Forest model are `ca` (number of major vessels colored by flourosopy), `thal` (thalassemia type), and `cp` (chest pain type).

This task successfully demonstrates the principles of decision trees, the risk of overfitting, and the power of ensemble methods like Random Forests to create more accurate and generalizable predictive models.
