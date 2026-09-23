# Interconnect Customer Churn Prediction

## Project Overview

Interconnect is a telecommunications operator that wants to reduce customer churn by identifying customers who are likely to cancel their services.

The goal of this project is to develop a machine learning classification model that can identify customers at risk of churn so that the company can proactively offer promotions, plan changes, or other retention strategies.

The project focuses primarily on **ROC-AUC**, with **Recall** as an additional business-oriented metric because identifying customers who actually leave is particularly important for retention campaigns.

---

## Business Problem

Customer churn represents a significant challenge for telecommunications companies.

Instead of waiting until customers cancel their contracts, Interconnect wants to identify customers who are more likely to leave and intervene beforehand.

The machine learning solution should therefore:

1. Identify customers at higher risk of churn.
2. Distinguish between customers who are likely to stay and those who are likely to leave.
3. Help the retention team prioritize customers for targeted interventions.

---

## Dataset

The data is divided into four sources:

### Contract Data

`contract.csv`

Contains information about:

* Contract type
* Billing method
* Payment method
* Monthly charges
* Total charges
* Contract start date
* Contract end date

### Personal Data

`personal.csv`

Contains customer information such as:

* Gender
* Senior citizen status
* Partner
* Dependents

### Internet Data

`internet.csv`

Contains information about Internet services:

* Internet service type
* Online security
* Online backup
* Device protection
* Technical support
* Streaming TV
* Streaming movies

### Phone Data

`phone.csv`

Contains information about:

* Multiple telephone lines

The datasets are connected through the unique customer identifier:

```text
customer_id
```

---

## Project Workflow

The complete workflow was:

```text
Raw Data
   │
   ▼
Data Loading
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Data Cleaning
   │
   ▼
Target Creation
   │
   ▼
Merge Four Data Sources
   │
   ▼
Feature Engineering
   │
   ├── tenure_months
   │
   ▼
Train / Validation / Test Split
   │
   ▼
One-Hot Encoding
   │
   ▼
Baseline Models
   │
   ├── Logistic Regression
   ├── LinearSVC
   ├── Random Forest
   └── LightGBM
   │
   ▼
Class Imbalance Experiments
   │
   ├── No balancing
   ├── Class weighting
   └── Oversampling
   │
   ▼
LightGBM Hyperparameter Tuning
   │
   ▼
5-Fold Stratified Cross-Validation
   │
   ▼
Final Model
   │
   ▼
TEST Evaluation
   │
   ▼
ROC-AUC = 0.941063
```

---

# Exploratory Data Analysis

The four datasets were first inspected individually to understand their structure, data types, missing values, duplicates, and distributions.

The datasets were then merged using `customer_id`, with the contract dataset used as the main table.

A one-to-one relationship was validated during the merge process to ensure that the integration did not unexpectedly duplicate customers.

The final dataset contained:

```text
7,043 customers
```

The target variable showed an imbalanced distribution:

```text
Active customers: approximately 73.5%
Churned customers: approximately 26.5%
```

This imbalance was considered during model development.

---

# Data Preprocessing

## Target Variable

The original contract information contained the customer's end date.

The target variable was created as:

```text
churn = 0 → customer remains active
churn = 1 → customer has churned
```

The end date itself was not used as a predictive feature because it directly contains information about the churn event and could introduce target leakage.

---

## Missing Values

Missing values generated during the merge were analyzed according to their business meaning.

For example, missing Internet service-related values represented customers who did not have Internet service.

These values were therefore transformed according to their meaning:

* Binary service variables → `0`
* `internet_service` → `"No internet"`

Missing values in `multiple_lines` were interpreted as customers without multiple phone lines and encoded accordingly.

---

# Feature Engineering

## Customer Tenure

A `tenure_months` feature was created to represent the approximate length of the customer's relationship with the company.

The feature was deliberately calculated using a common reference date rather than the customer's churn date.

This avoids directly using future churn information when estimating customer tenure.

The feature was tested both with and without the model.

The comparison showed that `tenure_months` provided a substantial improvement in predictive performance and was therefore retained in the final model.

---

# Categorical Encoding

The main categorical variables were:

```text
gender
type
payment_method
internet_service
```

They were transformed using One-Hot Encoding:

```python
OneHotEncoder(handle_unknown='ignore')
```

This transformation allowed categorical information to be represented numerically while also handling categories that could appear in validation or test data but were not observed during training.

---

# Train / Validation / Test Split

The data was divided using a stratified split:

```text
75% → Training
15% → Validation
10% → Test
```

Stratification was used to preserve the proportion of churned and active customers across the datasets.

The test set was kept separate throughout model selection and hyperparameter tuning.

---

# Baseline Models

Several classification algorithms were evaluated:

* Logistic Regression
* LinearSVC
* Random Forest
* LightGBM

The initial comparison showed that LightGBM achieved the strongest ROC-AUC performance.

Approximate validation ROC-AUC results were:

| Model               | Validation ROC-AUC |
| ------------------- | -----------------: |
| LightGBM            |         ~0.92–0.93 |
| Random Forest       |              ~0.86 |
| Logistic Regression |              ~0.84 |
| LinearSVC           |              ~0.83 |

These results motivated further optimization of the LightGBM model.

---

# Handling Class Imbalance

Three strategies were evaluated:

1. No class balancing.
2. Class weighting.
3. Oversampling of the minority class.

Oversampling produced the strongest ROC-AUC results for the selected LightGBM configuration.

The minority class was therefore oversampled during training.

Importantly, oversampling was applied only to training data and not to validation or test data.

During cross-validation, oversampling was also performed independently inside each training fold to prevent information leakage.

---

# Hyperparameter Optimization

LightGBM hyperparameters were progressively evaluated.

The final configuration selected for the model was:

```python
LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=75,
    random_state=42,
    verbosity=-1
)
```

The model was trained using:

* One-Hot Encoded features
* `tenure_months`
* Oversampling
* LightGBM

---

# Cross-Validation

Before performing the final test evaluation, the selected model was evaluated using **5-fold stratified cross-validation**.

Results:

|          Fold |      ROC-AUC |
| ------------: | -----------: |
|             1 |     0.922171 |
|             2 |     0.923144 |
|             3 |     0.925368 |
|             4 |     0.951454 |
|             5 |     0.929531 |
|      **Mean** | **0.930334** |
| **Std. Dev.** | **0.012142** |

The average ROC-AUC of **0.930334** indicates strong performance across different subsets of the training data.

The standard deviation of **0.012142** indicates that the model's performance remained relatively stable across folds.

---

# Final Model

After model selection and cross-validation, the final model was retrained using the combined training and validation datasets.

The test dataset remained untouched until this final evaluation.

---

# Final Test Results

The final model achieved:

| Metric      |  Test Result |
| ----------- | -----------: |
| **ROC-AUC** | **0.941063** |
| **Recall**  | **0.791444** |
| F1          |     0.800000 |
| Precision   |     0.808743 |
| Accuracy    |     0.895035 |

### ROC-AUC

The final **ROC-AUC of 0.941063** indicates that the model has a high ability to distinguish between customers who churn and customers who remain active.

This result is substantially above the project's minimum target of 0.75.

### Recall

The final Recall was:

```text
0.791444
```

This means that the model correctly identifies approximately **79.1% of customers who actually churn** using the selected classification threshold.

From a business perspective, this is particularly relevant because the retention team wants to identify customers before they leave.

---

# Business Impact

The model can be used as a customer-retention prioritization tool.

Instead of contacting every customer with the same strategy, Interconnect could use the model's churn probabilities to identify customers with higher estimated risk.

Potential actions include:

* Personalized retention offers.
* Discounts or temporary benefits.
* Contract or plan recommendations.
* Proactive customer support.
* Service-quality follow-ups.
* Targeted marketing campaigns.

The model should be considered a decision-support tool rather than a deterministic prediction system.

A Recall of approximately 79.1% also means that some customers who eventually churn will not be identified by the model at the selected threshold. Therefore, the model should be combined with business rules, customer history, campaign constraints, and other relevant information.

---

# Key Findings

### 1. Customer tenure was highly informative

The `tenure_months` feature produced a substantial improvement in model performance.

Models without this feature performed considerably worse than equivalent models using it.

### 2. LightGBM performed best

LightGBM consistently outperformed Logistic Regression, LinearSVC, and Random Forest in ROC-AUC.

### 3. Class imbalance mattered

Testing different strategies showed that oversampling the minority class improved the selected LightGBM configuration.

### 4. Cross-validation confirmed model stability

The model achieved:

```text
CV ROC-AUC = 0.930334 ± 0.012142
```

across five stratified folds.

### 5. The final test result exceeded the project target

The final model achieved:

```text
TEST ROC-AUC = 0.941063
```

which is well above the required minimum of 0.75.

---

# Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Machine Learning Techniques

* Exploratory Data Analysis
* Data Cleaning
* Feature Engineering
* Binary Classification
* One-Hot Encoding
* Stratified Train/Validation/Test Split
* Oversampling
* Hyperparameter Optimization
* Stratified K-Fold Cross-Validation
* ROC-AUC
* Recall
* Precision
* F1 Score
* Accuracy

---

# Project Structure

```text
interconnect-customer-churn/
│
├── data/
│   └── final_provider/
│       ├── contract.csv
│       ├── personal.csv
│       ├── internet.csv
│       └── phone.csv
│
├── notebooks/
│   └── telecom_churn_prediction.ipynb
│
├── README.md
├── requirements.txt
```

---

# Conclusion

The project successfully developed a machine learning solution for customer churn prediction at Interconnect.

The final LightGBM model achieved a **ROC-AUC of 0.941063 on the unseen test set** and a **Recall of 0.791444**.

The combination of feature engineering, categorical encoding, class balancing, model comparison, hyperparameter optimization, and cross-validation resulted in a model capable of effectively distinguishing customers at higher risk of churn.

The model can therefore serve as a foundation for a proactive customer-retention strategy, allowing Interconnect to prioritize customers according to their estimated churn risk and potentially intervene before cancellation occurs.

---

# Reproducibility

The project is designed to be fully reproducible from the provided notebook and dependency file.

The notebook can be executed from beginning to end using **Run All** without requiring manual intervention between sections.

## Requirements

The project dependencies are listed in:

```text
requirements.txt
```

The project does not require an `environment.yml` file.

The main libraries used in the project include:

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Jupyter Notebook

## Installation

Create and activate a Python environment, then install the required dependencies:

```bash
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the project notebook:

```text
notebooks/telcom_churn_prediction.ipynb
```

Make sure the required dataset files are available under:

```text
data/final_provider/
```

The notebook is structured so that all preprocessing, feature engineering, model training, validation, cross-validation, and final evaluation are executed sequentially.

To reproduce the complete analysis:

1. Install the dependencies from `requirements.txt`.
2. Open the notebook in Jupyter Notebook, JupyterLab, or VS Code.
3. Verify that the dataset paths point to the files in `data/final_provider/`.
4. Run the notebook using **Run All**.

No intermediate notebook outputs or manually generated variables are required before execution.

## Reproducibility Considerations

Random seeds were explicitly defined in the main train/validation/test split, oversampling procedure, cross-validation, and LightGBM model.

The main random seed used throughout the project is:

```python
random_state=54321
```

and the final LightGBM model uses:

```python
random_state=42
```

These settings help reproduce the same data partitions and model results when the same software environment and dataset are used.

## Expected Final Result

After running the complete notebook, the final model should produce performance close to the reported test results:

| Metric    | Expected Test Result |
| --------- | -------------------: |
| ROC-AUC   |         **0.941063** |
| Recall    |         **0.791444** |
| F1        |         **0.800000** |
| Precision |         **0.808743** |
| Accuracy  |         **0.895035** |

The model's 5-fold cross-validation result was:

```text
Mean ROC-AUC: 0.930334
Standard deviation: 0.012142
```

Small differences in execution time or floating-point results may occur depending on the Python version, package versions, operating system, and hardware.

The reported metrics correspond to the original project execution environment and dataset.
