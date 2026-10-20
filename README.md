# Customer Churn Prediction Using CatBoost

## Project Overview
This project uses machine learning to predict customer churn for a telecommunications company. The goal is to identify customers who may leave the service and help businesses make data-driven retention decisions.

## Business Problem
Customer churn can reduce revenue and increase customer acquisition costs. Predicting potential churners helps businesses prioritize retention efforts and improve customer loyalty.

## Dataset
- **Source:** [IBM Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7,043 customers
- **Original features:** 21 columns
- **Model input features:** 19 columns after removing the customer ID and separating the target
- **Target variable:** `Churn`

## Tools and Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- CatBoost
- Jupyter Notebook

## Project Workflow
1. Loaded and inspected the dataset.
2. Checked data quality and handled missing `TotalCharges` values.
3. Performed exploratory data analysis (EDA).
4. Prepared features and encoded the target variable.
5. Split the data into training and testing sets.
6. Trained a CatBoost classification model.
7. Evaluated performance using classification metrics and a confusion matrix.
8. Analyzed missed churn cases.
9. Saved and reloaded the trained model to verify its predictions.

## Model Performance

Evaluation on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 67.12% |
| Recall | 52.41% |
| F1-score | 58.86% |
| ROC-AUC | 84.69% |

### Confusion Matrix

| | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 939 | 96 |
| Actual: Churn | 178 | 196 |

The model correctly identified 196 customers who churned, while 178 actual churners were missed at the selected classification threshold.

## Key Findings
- Customer contract type and tenure were useful areas to investigate during churn analysis.
- Missed churn cases were concentrated among month-to-month customers.
- The model's recall indicates that some customers who churned were not identified, so further tuning may be valuable.

## Saved Model
The trained CatBoost model is included as `customer_churn_catboost.cbm`.

The model was saved, reloaded, and checked to confirm that its predictions matched the original model on the test data.

## How to Run
1. Download or clone this repository.
2. Install the required Python libraries.
3. Open `customer_churn_prediction.ipynb` in Jupyter Notebook or Google Colab.
4. Run the notebook cells in order.

The notebook downloads the dataset using KaggleHub. You may need to configure Kaggle access in your environment.

## Limitations and Future Improvements
- Improve churn recall through threshold tuning and class-imbalance strategies.
- Compare CatBoost with other classification models.
- Tune hyperparameters using cross-validation.
- Further evaluate model calibration and business costs of false negatives.
- Validate performance on new, unseen data before using it in production.

## Project Type
Portfolio project for machine learning and data analysis practice.

## Author
**Mian Semab**

GitHub: [@miansemab](https://github.com/miansemab)
