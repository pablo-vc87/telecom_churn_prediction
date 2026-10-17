# Telecom Customer Churn Prediction

> Versión en español: [LEEME.md](LEEME.md)

A binary classification project that predicts **which customers of a telecom operator are likely to cancel their service (churn)**, so that the company can offer them promotions before they leave.

The project integrates four data sources, builds a leakage-free **customer tenure** feature, and compares **12 model configurations** (4 models × 3 class-imbalance strategies) before tuning the best one. The selected model, **LightGBM with oversampling of the minority class**, achieved a **mean ROC-AUC of 0.9303 in 5-fold cross-validation** and **0.9411 ROC-AUC with 0.79 Recall on the independent test set**, well above the requirement of **ROC-AUC ≥ 0.75**.

---

# Business Context

Interconnect, a telecom operator, wants to reduce customer churn. If the company can identify in advance the customers who are about to leave, the marketing team can offer them promotions or special plans. A reliable model helps to:

* Focus retention campaigns on the customers with the highest risk.
* Reduce the cost of offering discounts to customers who would stay anyway.
* Understand which services and contract conditions are associated with churn.

The model must predict one target:

| Target | Description |
| --- | --- |
| `churn` | 1 if the customer cancelled the service (`EndDate` is a date), 0 if the customer is still active (`EndDate = "No"`) |

---

# Churn Modeling Concepts

| Term | Meaning |
| --- | --- |
| Churn | A customer who cancels the service |
| Class imbalance | Only about 26.5% of customers churned, so a model can look accurate by predicting "no churn" for everyone |
| Data leakage | Using information that would not be available at prediction time (for example, the cancellation date) |
| Tenure | How long a customer has been with the company |
| `class_weight` / `scale_pos_weight` | Giving more weight to the minority class during training |
| Oversampling | Repeating minority-class rows in the training set so both classes are balanced |
| Stratified split | Keeping the same churn proportion in train, validation, and test |

## Feature Groups

```text
Contract:   type, paperless_billing, payment_method, monthly_charges, total_charges
Personal:   gender, senior_citizen, partner, dependents
Internet:   internet_service, online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies
Phone:      multiple_lines
Engineered: tenure_months
```

---

# Evaluation Metric

The main metric is **ROC-AUC** (area under the ROC curve). It measures how well the model **ranks** churners above non-churners across all possible classification thresholds, so it does not depend on the chosen threshold and is robust to class imbalance.

**Recall** is used as a secondary metric because, for retention, missing a customer who leaves is more costly than contacting one who would have stayed:

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

The project requirement was **ROC-AUC ≥ 0.75** on the test set. F1, Precision, and Accuracy were also recorded to interpret each model.

---

# Dataset

The data come from four files linked by `customerID`.

| File | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| `datasets/final_provider/contract.csv` | 7,043 | 8 | Contract type, billing, payment method, charges, start and end dates |
| `datasets/final_provider/personal.csv` | 7,043 | 5 | Customer demographics |
| `datasets/final_provider/internet.csv` | 5,517 | 8 | Internet services (DSL or fiber optic, security, backup, streaming…) |
| `datasets/final_provider/phone.csv` | 6,361 | 2 | Phone service (multiple lines) |

* **7,043 customers** in total: 1,526 without internet service and 682 without phone service.
* Customers started between **October 2013 and February 2020**; all cancellations happened between **October 2019 and January 2020**.
* Churn rate: **1,869 customers (26.5%)** churned and **5,174 (73.5%)** stayed.

**Data source.** The datasets were provided by **TripleTen** as part of its Data Science Bootcamp and are included in this repository for educational and reproducibility purposes.

---

# Project Workflow

```text
Raw data (4 files)
            │
            ▼
1. Data preparation
   ├── snake_case column names, dates, numeric total_charges
   ├── Target: churn = EndDate is not "No"
   └── Left joins on customer_id + missing values from the joins
            │
            ▼
2. Feature engineering: tenure_months (without future information)
            │
            ▼
3. Stratified split: 75% train / 15% validation / 10% test + One-Hot Encoding
            │
            ▼
4. Modeling
   ├── Experiment 1: 4 models × 3 imbalance strategies (12 configurations)
   ├── Experiment 2: LightGBM with vs without tenure_months
   ├── Experiment 3: LightGBM hyperparameter searches
   ├── 5-fold stratified cross-validation (oversampling inside each fold)
   └── Retrain on train + validation and evaluate once on test
```

---

# 1. Data Preparation

* Column names were converted to `snake_case`.
* `begin_date` and `end_date` were parsed as dates (format `YYYY-MM-DD`). `end_date = "No"` means the customer is still active.
* `total_charges` had **11 empty values**, all from customers who signed up on February 1, 2020 and had not received their first bill yet. They were set to **0**.
* "Yes"/"No" columns were converted to 1/0.
* The four tables were joined to `contract` with left joins on `customer_id`, checking one-to-one relationships (`validate='one_to_one'`).
* Missing values created by the joins mean that the customer does not have that service:
  * Internet service columns → **0**, and `internet_service` → **"No internet"** (1,526 customers).
  * `multiple_lines` → **0** (682 customers without phone service).
* `customer_id`, `begin_date`, `end_date`, and `churn` were excluded from the features.

---

# 2. Feature Engineering: Customer Tenure

`tenure_months` estimates how long each customer has been with the company. It was calculated **without using the cancellation date**:

| Option | What it represents | Usable? |
| --- | --- | --- |
| `end_date − begin_date` | Final length of the relationship, known only after the customer leaves | ❌ Future information |
| `end_date` | When the churn happened | ❌ Direct leakage |
| `cutoff_date − begin_date` | Tenure up to a common cutoff date | ✅ |

The cutoff date is the latest `begin_date` in the data (**2020-02-01**):

```python
fecha_corte = df['begin_date'].max()
df['tenure_months'] = (fecha_corte - df['begin_date']).dt.days / 30.44
```

The resulting feature ranges from **0 to 76 months**, with a mean of **33.1 months**.

---

# 3. Data Split and Encoding

The data were split with `stratify` so that every set keeps the 26.5% churn rate:

```text
100% of the data (7,043 customers)
│
├── 75% TRAIN ─────────→ 5,282 customers → model training
├── 15% VALIDATION ────→ 1,056 customers → model and hyperparameter selection
└── 10% TEST ──────────→   705 customers → final evaluation (used only once)
```

The categorical variables (`gender`, `type`, `payment_method`, `internet_service`) were encoded with `OneHotEncoder(handle_unknown='ignore')` inside a `ColumnTransformer`, fitted **only on the training set**. The final matrix has **26 features**.

For oversampling, the churn rows of the training set were repeated 3 times (from 1,402 to 4,206 churners vs 3,880 non-churners). Validation and test were never modified.

---

# 4. Modeling

## Experiment 1: Models and Imbalance Strategies

Four models were trained with three strategies each: **no balancing**, **class weights** (`class_weight='balanced'`, or `scale_pos_weight ≈ 2.77` for LightGBM), and **oversampling**.

| Model | No balancing | Class weight | Oversampling |
| --- | ---: | ---: | ---: |
| **LightGBM** | 0.9239 | 0.9201 | **0.9274** |
| Random Forest | 0.8607 | 0.8581 | 0.8607 |
| Logistic Regression | 0.8344 | 0.8339 | 0.8337 |
| LinearSVC | 0.8320 | 0.8333 | 0.8328 |

*Validation ROC-AUC.*

* **LightGBM** was clearly the best model, which suggests non-linear relationships between the features and churn.
* Balancing **increased Recall** in every model (for Logistic Regression, from about 0.54 to 0.77) at the cost of lower Precision, while **ROC-AUC barely changed**: balancing moves the decision threshold, but the ranking of customers stays almost the same.
* Accuracy was not used as the main metric: the unbalanced models had the highest Accuracy because they mostly predicted the majority class.

## Experiment 2: Impact of `tenure_months`

| LightGBM (oversampling) | Validation ROC-AUC |
| --- | ---: |
| With `tenure_months` | **0.9274** |
| Without `tenure_months` | 0.8256 |

Removing tenure reduced ROC-AUC by about **0.10**, so the feature was kept.

## Experiment 3: Hyperparameter Tuning (LightGBM + oversampling)

| Search | What changed | Best configuration | Best Validation ROC-AUC |
| --- | --- | --- | ---: |
| 1 | `n_estimators` (100–300) and `num_leaves` (15, 31, 63), `learning_rate = 0.05` | 300 trees, 31 leaves | 0.9314 |
| 2 | `min_child_samples` from 10 to 100, with 300 trees and 63 leaves | `min_child_samples = 75` | 0.9355 |
| 3 | `min_child_samples` from 40 to 90 around the best value | `min_child_samples = 75` | 0.9355 |

In search 1, 31 leaves gave a slightly higher ROC-AUC than 63 leaves (0.9314 vs 0.9296). The second search kept **63 leaves** because the larger trees can be regularized with `min_child_samples`; with `min_child_samples = 75` they surpassed the first search (0.9355).

Final configuration:

```text
LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
               max_depth=-1, min_child_samples=75, random_state=42)
```

## Cross-Validation

Before using the test set, the final configuration was checked with **5-fold stratified cross-validation** on the training set. Oversampling was applied **inside each fold**, only on the fold's training part, so no duplicated row appears in both training and validation.

| Fold | ROC-AUC |
| ---: | ---: |
| 1 | 0.9222 |
| 2 | 0.9231 |
| 3 | 0.9254 |
| 4 | 0.9515 |
| 5 | 0.9295 |
| **Mean ± std** | **0.9303 ± 0.0121** |

## Final Result

The model was retrained on **train + validation** (with oversampling) and evaluated **once** on the test set:

```text
Model:          LightGBM + oversampling
CV ROC-AUC:     0.9303 ± 0.0121
Test ROC-AUC:   0.9411
Test Recall:    0.7914
Test F1:        0.8000
Test Precision: 0.8087
Test Accuracy:  0.8950
Requirement:    ROC-AUC ≥ 0.75   ✓
```

On the 705 test customers, the model identified about **79% of the 187 customers who churned**, and about 81% of the customers it flagged as churners actually left.

---

# Conclusions

1. **Integrating the four sources required interpreting missing values correctly**: a customer missing from the internet or phone table simply does not have that service.

2. **`tenure_months` was the most informative feature.** Building it with a common cutoff date, instead of the cancellation date, avoided data leakage; removing it lowered ROC-AUC from 0.927 to 0.826.

3. **LightGBM clearly outperformed the other models** (validation ROC-AUC around 0.92–0.93 vs 0.86 for Random Forest and 0.83 for the linear models).

4. **Balancing the classes mainly changed Recall, not ROC-AUC.** Oversampling gave the best ROC-AUC for LightGBM and was kept for the final model.

5. **The result is stable and above the requirement**: cross-validation gave 0.9303 ± 0.0121 and the test set gave **0.9411 ROC-AUC with 0.79 Recall**.

## Business Interpretation

With a Recall of about 79%, the model detects roughly 79 of every 100 customers who are going to leave, so the marketing team can prioritize them for offers, plan changes, or a direct call. About 21% of churners would still be missed with the current threshold, so the model should support retention decisions, not replace them.

## Next Steps

* Tune the classification threshold according to the cost of a retention offer versus the cost of losing a customer.
* Analyze feature importance to explain which contract conditions and services drive churn.
* Note: an earlier version of the notebook parsed the dates as `YYYY-DD-MM`. The current version uses the correct format; the LightGBM results did not change, because the error preserved the order of `tenure_months`.

---

# Project Structure

```text
telecom_churn_prediction/
│
├── datasets/
│   └── final_provider/
│       ├── contract.csv
│       ├── internet.csv
│       ├── personal.csv
│       └── phone.csv
│
├── notebooks/
│   └── telecom_churn_prediction.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/pablo-vc87/telecom_churn_prediction.git

cd telecom_churn_prediction
```

## Option 1: Conda

```bash
conda env create -f environment.yml
conda activate telecom_churn_prediction
jupyter notebook
```

## Option 2: pip

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/telecom_churn_prediction.ipynb` and run all cells.

---

# Technologies

* Python
* Pandas
* NumPy
* scikit-learn
* LightGBM
* Jupyter Notebook

---

# Skills Demonstrated

* Integration of multiple data sources
* Interpretation of missing values after joins
* Leakage-free feature engineering (customer tenure)
* Binary classification with class imbalance
* Class weights and oversampling
* One-Hot Encoding with `ColumnTransformer`
* Model comparison: Logistic Regression, LinearSVC, Random Forest, and LightGBM
* Hyperparameter tuning
* Stratified cross-validation with oversampling inside the folds
* ROC-AUC, Recall, Precision, and F1 evaluation

---

## Author

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
