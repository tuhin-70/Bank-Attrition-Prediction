# Proactive Bank Customer Churn Prediction

## 1. Overview
This project develops a machine learning solution to proactively identify customers at risk of churning (closing their credit card accounts) for a financial institution. By leveraging advanced predictive modeling techniques, the solution enables targeted retention strategies that significantly reduce customer attrition and increase profitability.

## 2. The "Why": Prioritizing Recall for Business Impact
In churn prediction, not all model errors are equal.

- **False Positive:** Predicting a customer will churn, but they stay.
  - *Business Cost:* A small, unnecessary marketing expense (e.g., a retention offer).

- **False Negative:** Predicting a customer will stay, but they churn.
  - *Business Cost:* **Loss of the customer's entire future lifetime value.** This is far more damaging.

Our primary objective was to minimize False Negatives. Therefore, the models were systematically optimized to maximize **Recall**. A high recall score ensures that we correctly identify the vast majority of at-risk customers, which is the most critical goal for an effective retention strategy.

## 3. Performance of the Final Model
After evaluating multiple classifiers—including Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting—the final, tuned **XGBoost model combined with SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance delivered the best performance.

The model's performance on the unseen test set was:

- **Test Accuracy: 96%**
- **Recall (for the Churn class): 90%**

This strong **90% recall** means the model successfully identified 90 out of every 100 customers who were actually going to churn, giving the bank a powerful tool to prevent revenue loss.

## 4. Financial Impact Calculation

| Metric | Value |
|--------|-------|
| Customers correctly identified as at-risk (True Positives) | 288 |
| Successful retentions (25% success rate) | 72 customers |
| Value of retained customers | $32,400 |
| Cost of retention campaign | $16,550 |
| **Per-customer value** | **$7.82** |

## 5. Technical Workflow
1. **Data Cleaning & Preprocessing:** Standardized feature names, handled missing values by creating explicit "Missing" categories, and removed redundant columns identified through correlation analysis.
2. **Exploratory Data Analysis (EDA):** Uncovered key churn indicators, such as lower transaction counts (Total_Trans_Ct), lower transaction amounts (Total_Trans_Amt), and higher inactivity (Months_Inactive_12_mon).
3. **Preprocessing Pipeline:** Built a robust scikit-learn pipeline to handle scaling (RobustScaler), ordinal encoding, and one-hot encoding consistently across training and test sets.
4. **Handling Class Imbalance:** Integrated **SMOTE** into the modeling pipeline to create a balanced training environment, which was critical for improving recall.
5. **Model Training & Tuning:** Systematically trained and evaluated multiple classification models. Used GridSearchCV to fine-tune hyperparameters, with **recall** as the primary optimization metric.
6. **Model Selection:** The tuned **SMOTE + XGBoost** model was selected for its superior ability to identify churners correctly.

## 6. Repository Contents
- `bank-churn-prediction.ipynb`: Complete Jupyter notebook with EDA, modeling, and evaluation
- `BankChurners.csv`: Dataset used for training and evaluation
- `best_xgb_smote_model.joblib`: The final, saved, and trained model pipeline
- `README.md`: This documentation file

## 7. Tech Stack
- **Programming Language:** Python 3
- **Libraries:**
  - Data Manipulation & Analysis: Pandas, NumPy
  - Data Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, XGBoost
  - Imbalanced Data Handling: Imbalanced-learn (for SMOTE)
