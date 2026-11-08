# ML Assignment 2 - Classification Models Comparison

## a. Problem Statement
The objective of this assignment is to implement and compare five different machine learning classification models on a tabular dataset. Furthermore, the models are deployed in an interactive Streamlit web application where users can upload test data to evaluate the models without retraining them.

## b. Dataset Description
- **Dataset Name:** Dresses Attribute Sales
- **Source:** UCI Machine Learning Repository / OpenML
- **Instances:** 500
- **Features:** 12 input features (Style, Price, Rating, Size, Season, NeckLine, SleeveLength, waiseline, Material, FabricType, Decoration, Pattern Type) and 1 target column (Recommendation).
- **Preprocessing:** Handled missing values (imputed with mode), label-encoded categorical variables, and applied standard scaling for distance-based and linear models (Logistic Regression, KNN, SVM).

## c. GitHub Repository Link
*[Insert your GitHub repository link here once you push the code]*

## d. Models Used and Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.5700 | 0.5378 | 0.4800 | 0.2857 | 0.3582 | 0.0702 |
| Decision Tree | 0.5800 | 0.5722 | 0.5000 | 0.5238 | 0.5116 | 0.1437 |
| KNN | 0.5400 | 0.5025 | 0.4474 | 0.4048 | 0.4250 | 0.0434 |
| Naive Bayes | 0.5600 | 0.5402 | 0.4615 | 0.2857 | 0.3529 | 0.0499 |
| Support Vector Machine | 0.5400 | 0.5337 | 0.4375 | 0.3333 | 0.3784 | 0.0243 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Performed moderately with an accuracy of 57%. It assumes a linear decision boundary, which may not perfectly capture the complex relationships in mostly categorical data. |
| Decision Tree | Achieved the highest overall performance across all metrics (Accuracy: 58%, MCC: 0.14). It handles categorical splits well without requiring scaling. |
| KNN | Yielded lower performance (54%). The dataset contains mostly categorical features converted to numerical codes, making distance-based metrics less effective for KNN. |
| Naive Bayes | Achieved 56% accuracy. The assumption of feature independence is likely violated here, leading to moderate but not top-tier performance. |
| Support Vector Machine | Achieved 54% accuracy. The model struggled to find a clear maximum-margin hyperplane, likely due to overlapping classes in the feature space. |
| **Overall Winner** | **Decision Tree** performed the best on this dataset, yielding the highest Accuracy, AUC, Precision, F1, and MCC scores. |
