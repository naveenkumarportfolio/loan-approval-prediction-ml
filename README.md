# Loan Approval Prediction using Machine Learning

A machine learning project that predicts loan approval using applicant information and compares the performance of three classification algorithms:

- Decision Tree
- Naive Bayes
- K-Nearest Neighbors (KNN)

The project evaluates the models using Accuracy, Precision, Recall, F1-score and Confusion Matrix.

---

## Project Objective

The main business question is:

> Can applicant information be used to predict whether a loan application will be approved?

The project uses historical loan application data to train classification models and compare their performance.

---

## Dataset

The dataset contains 614 loan applications and 13 columns, including the target variable and applicant ID.

### Target Variable

`Loan_Status`

- `Y` = Loan Approved
- `N` = Loan Rejected

### Features

#### Numerical Features

- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term

#### Categorical Features

- Gender
- Married
- Dependents
- Education
- Self_Employed
- Credit_History
- Property_Area

`Loan_ID` is removed because it is an identifier rather than a predictive feature.

---

## Data Preprocessing

The following preprocessing steps are performed:

1. Load and inspect the dataset
2. Remove duplicate rows
3. Remove `Loan_ID`
4. Convert the target variable:
   - Y → 1
   - N → 0
5. Convert `Credit_History` into categorical values:
   - 1.0 → Good
   - 0.0 → Bad
6. Handle missing numerical values using median imputation
7. Handle missing categorical values using most-frequent imputation
8. Apply one-hot encoding to categorical variables
9. Standardize numerical variables
10. Split the data using an 80/20 stratified train-test split

The preprocessing steps are included inside the machine learning pipeline to reduce the risk of data leakage.

---

## Machine Learning Models

### 1. Decision Tree

A Decision Tree uses if-then decision rules to classify loan applications.

Parameters used:

```text
max_depth = 4
min_samples_leaf = 5
2. Naive Bayes

Gaussian Naive Bayes is used as a probability-based classification model.

3. K-Nearest Neighbors

KNN classifies an applicant based on similarity to nearby observations.

Parameters used:

n_neighbors = 5
