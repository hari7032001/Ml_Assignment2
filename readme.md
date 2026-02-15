# Machine Learning Assignment 2  
## Bank Marketing Classification

---

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether a customer will subscribe to a term deposit based on the Bank Marketing dataset.

The task is a binary classification problem where the target variable indicates whether the client subscribed to a term deposit (yes/no).

---

## b. Dataset Description

Dataset Used: Bank Marketing Dataset (bank-full.csv)

Source: UCI Machine Learning Repository

Minimum Feature Size: 16 features  
Minimum Instance Size: 45,211 records  

The dataset contains demographic and campaign-related information of customers contacted during direct marketing campaigns conducted by a Portuguese banking institution.

Target Variable:
- y = yes (1)
- y = no (0)

The dataset contains both categorical and numerical features such as:
- age
- job
- marital status
- education
- balance
- housing loan
- previous campaign outcome
- etc.

Preprocessing Steps:
- Converted target labels to binary (yes = 1, no = 0)
- Encoded categorical variables using Label Encoding
- Standardized features using StandardScaler
- Split dataset into 80% training and 20% testing data using stratified sampling

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.8897 | 0.8569 | 0.5864 | 0.1957 | 0.2934 | 0.2943 |
| Decision Tree | 0.8784 | 0.7081 | 0.4804 | 0.4858 | 0.4831 | 0.4142 |
| KNN | 0.8811 | 0.7527 | 0.4860 | 0.2788 | 0.3544 | 0.3080 |
| Naive Bayes | 0.8445 | 0.8160 | 0.3659 | 0.4490 | 0.4032 | 0.3171 |
| Random Forest (Ensemble) | 0.9063 | 0.9243 | 0.6586 | 0.4140 | 0.5084 | 0.4748 |
| XGBoost (Ensemble) | 0.9058 | 0.9267 | 0.6281 | 0.4773 | 0.5424 | 0.4968 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Although it achieved high accuracy (0.8897), the very low recall (0.1957) shows that the model struggles to correctly identify positive class instances, favoring the majority class instead. |
| Decision Tree | Demonstrated a relatively balanced precision (0.4804) and recall (0.4858), leading to a moderate F1 score (0.4831). However, its lower AUC (0.7081) indicates weaker overall discrimination capability compared to ensemble models. |
| KNN | Achieved moderate accuracy (0.8811) but low recall (0.2788) and F1 score (0.3544), suggesting limited effectiveness in capturing minority class cases. |
| Naive Bayes | Recorded lower accuracy (0.8445), but comparatively better recall (0.4490) than Logistic Regression and KNN. However, lower precision (0.3659) reduced its overall predictive strength. |
| Random Forest (Ensemble) | Delivered the highest accuracy (0.9063) and strong AUC (0.9243), along with solid F1 (0.5084) and MCC (0.4748), indicating strong generalization and balanced performance. |
| XGBoost (Ensemble) | Achieved the highest AUC (0.9267), F1 score (0.5424), and MCC (0.4968), with the strongest recall (0.4773) among the top-performing models, making it the most effective overall for this imbalanced dataset. |

---

## Conclusion

Based on the comparative analysis of all models, ensemble methods outperformed individual classifiers on the imbalanced Bank Marketing dataset. While Logistic Regression and KNN achieved relatively high accuracy, their low recall values indicate poor detection of positive class instances. Decision Tree and Naive Bayes showed moderate balance but lacked strong overall discrimination power. Among all models, XGBoost demonstrated the best overall performance, achieving the highest AUC (0.9267), F1-score (0.5424), and MCC (0.4968), along with strong recall. Therefore, XGBoost is identified as the most suitable model for term deposit prediction, as it provides the best balance between classification accuracy and minority class detection.

---