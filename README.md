# Loan Approval Prediction – Machine Learning Model Comparison

Task 02 of the **Devixo Solutions AI/ML Internship**. A lender has to decide quickly whether an application is safe to approve. This project builds six classifiers that learn from past applications and predict Approved or Rejected, then compares them side by side.

## Dataset

- **Source:** [Loan Approval Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)
- **Size:** 4,269 applications, 13 columns (no missing values, no duplicates)
- **Target:** `loan_status` (Approved = 1, Rejected = 0)
- **Features:** dependents, education, self-employment, annual income (`income_annum`), loan amount, loan term, CIBIL score, and residential, commercial, luxury and bank asset values

## Workflow

1. **Cleaning:** strip stray spaces from column names and text values, drop `loan_id`, encode the target
2. **Feature engineering:** `total_assets` (sum of the four asset columns) and `loan_to_income` (loan amount ÷ annual income)
3. **Preprocessing:** stratified 75/25 split (3,201 train / 1,068 test); a `ColumnTransformer` scales numeric columns and one-hot encodes `education` and `self_employed` inside a Scikit-learn `Pipeline`
4. **Models:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost
5. **Evaluation:** Accuracy, Precision, Recall, F1, confusion matrices, stratified 5-fold cross-validation, training time
6. **Bonus:** `RandomizedSearchCV` tuning of XGBoost, plus Linear Regression to estimate loan amount

## Results

| Model | Accuracy | Precision | Recall | F1 | CV Acc. | Train time (s) |
|---|---|---|---|---|---|---|
| **Random Forest** | **0.9953** | **0.9925** | **1.0000** | **0.9962** | 0.9959 | 1.2035 |
| Decision Tree | 0.9944 | 0.9925 | 0.9985 | 0.9955 | 0.9922 | 0.0303 |
| XGBoost | 0.9934 | 0.9910 | 0.9985 | 0.9947 | 0.9944 | 0.2097 |
| SVM | 0.9560 | 0.9611 | 0.9684 | 0.9647 | 0.9491 | 0.1463 |
| Logistic Regression | 0.9251 | 0.9465 | 0.9322 | 0.9393 | 0.9147 | 0.0394 |
| KNN | 0.9223 | 0.9278 | 0.9488 | 0.9382 | 0.9182 | 0.0167 |

**Best model:** Random Forest. It missed no genuine approvals and made only 5 wrong approvals out of 404 rejected applications. Decision Tree and XGBoost are almost as good, and all three tree-based models clearly beat the linear, distance-based and margin-based models, because credit score and the loan-to-income ratio drive the decision through threshold-style splits.

**Hyperparameter tuning:** best XGBoost settings were `n_estimators=100`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.7`, `colsample_bytree=0.7`. Test F1 stayed at 0.9947, but the tuned model is much simpler for the same score.

**Linear Regression (loan amount):** MAE 2,481,564, RMSE 3,285,205, R² 0.870.

## Limitations

- Scores near 99% suggest the data has very clear patterns, mainly driven by credit score. Real-world lending data is usually noisier.
- The data is historical, so any bias in past decisions may be learned by the model. Fairness checks are needed before real use.
- The model should support, not replace, a human credit officer.

## How to Run

1. Open `Task02_Loan_Approval_Prediction.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells (`Runtime` → `Run all`). The dataset downloads automatically through `kagglehub`; if that fails, upload the CSV when prompted.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn (Pipelines, ColumnTransformer), XGBoost, Matplotlib, Seaborn, Google Colab

## Project Structure

```
├── Task02_Loan_Approval_Prediction.ipynb   # full notebook with outputs
└── README.md
```

## Author

**Abdul Rafay**
