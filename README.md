Loan Approval Prediction using Machine Learning

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

ssian Naive Bayes is used as a probability-based classification model.

3. K-Nearest Neighbors

KNN classifies an applicant based on similarity to nearby observations.

Parameters used:

n_neighbors = 5
Experimental Setup

The models use the same train-test split for a consistent comparison.

Training data: 80%
Testing data: 20%
Random state: 42
Split type: Stratified

The dataset contains:

614 applications
491 training observations
123 testing observations
Model Performance
Model	Accuracy	Precision	Recall	F1-score	False Positives	False Negatives
Decision Tree	82.11%	84.62%	90.59%	87.50%	14	8
Naive Bayes	85.37%	83.84%	97.65%	90.22%	16	2
KNN	82.93%	81.37%	97.65%	88.77%	19	2
Results

In this experiment, Naive Bayes achieved:

Accuracy: 85.37%
Precision: 83.84%
Recall: 97.65%
F1-score: 90.22%

Based on the project's selection rule, the model with the highest F1-score for the Approved class is selected, with accuracy used as a tie-breaker.

However, model selection depends on the business objective.

For example, the Decision Tree produced fewer false positives, while Naive Bayes and KNN produced fewer false negatives.

Therefore, accuracy alone should not be used to evaluate a loan approval model.

Business Interpretation

There are two important types of prediction errors:

False Positive

The model predicts that a loan should be approved when the actual outcome is rejection.

This can represent a potentially risky approval.

False Negative

The model predicts rejection when the actual outcome is approval.

This can represent a potentially creditworthy applicant being rejected.

The relative importance of these errors depends on the bank's risk policy and business objectives.

Project Outputs

Running the Python script generates:

Accuracy comparison chart
Precision comparison chart
Recall comparison chart
F1-score comparison chart
Decision Tree confusion matrix
Naive Bayes confusion matrix
KNN confusion matrix

All generated charts are saved inside the output/ directory.

How to Run the Project
Step 1: Clone the repository
git clone https://github.com/YOUR-USERNAME/loan-approval-prediction-ml.git
Step 2: Open the project folder
cd loan-approval-prediction-ml
Step 3: Install dependencies
pip install -r requirements.txt
Step 4: Run the model
python loan_prediction.py

The program will create an output folder containing the generated charts.

Requirements

The project uses:

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

See requirements.txt for the required packages.

Limitations

This project is an academic machine learning experiment and should not be treated as a production credit-scoring system.

Important limitations include:

Results are based on a single train-test split.
Model performance may change with another split.
The dataset has class imbalance.
Accuracy alone does not capture all business risks.
False positives and false negatives have different financial implications.
Additional validation and cross-validation would be appropriate before real-world deployment.
Future Improvements

Possible improvements include:

Cross-validation
Hyperparameter tuning
ROC-AUC comparison
Precision-Recall curves
Feature importance analysis
Model explainability using SHAP
Testing additional classification algorithms
Handling class imbalance
Threshold optimization
Deployment using Streamlit or Flask
Author

Naveen

BBA – Finance & Marketing Analytics
CHRIST (Deemed to be University), Delhi NCR
