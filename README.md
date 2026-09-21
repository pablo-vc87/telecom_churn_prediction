## Statistical overview — Contract data

The `df_contract` dataset contains **7,043 customers** and information about their contracts, billing, start and end dates, monthly charges, and cumulative charges.

### Customer start and end dates

The `begin_date` variable contains a date for all 7,043 customers, ranging from **2013 to 2020**. The median start date is around January 2017, indicating that the dataset contains customers with substantially different contract tenures.

The `end_date` variable contains dates for **1,869 customers**, while the remaining customers have no recorded end date because they were still active at the time represented by the dataset.

The number of customers with an `end_date` is consistent with the target variable `churn`, where:

* **1,869 customers (26.54%)** have `churn = 1`.
* **5,174 customers (73.46%)** have `churn = 0`.

Therefore, the target variable presents a moderate class imbalance, which will be addressed later during the model-development stage.

### Monthly charges

`monthly_charges` is a numerical variable ranging from **18.25 to 118.75**.

| Statistic          | Value |
| ------------------ | ----: |
| Mean               | 64.76 |
| Median             | 70.35 |
| Q1                 | 35.50 |
| Q3                 | 89.85 |
| Standard deviation | 30.09 |

The difference between the mean and median suggests that the distribution is not perfectly symmetric. Nevertheless, the variable has a well-defined numerical range and will be retained as a continuous predictor.

### Total charges

`total_charges` ranges from **0 to 8,684.80**.

| Statistic          |    Value |
| ------------------ | -------: |
| Mean               | 2,279.73 |
| Median             | 1,394.55 |
| Q1                 |   398.55 |
| Q3                 | 3,786.60 |
| Standard deviation | 2,266.79 |

The mean is considerably higher than the median, indicating a right-skewed distribution. This is consistent with the fact that customers have different lengths of service: customers who have been subscribed for longer periods can accumulate substantially higher total charges.

The original dataset contained empty strings in `total_charges`. These records correspond to newly registered customers who had not yet received their first bill and therefore had not made a payment. These observations should not automatically be interpreted as erroneous data or as zero spending.

`total_charges` will therefore be retained as a potential predictor. Its relationship with customer tenure and churn should be examined before model training.

### Paperless billing

`paperless_billing` is a binary variable encoded as `0/1`.

Its mean is **0.5922**, indicating that approximately **59.2% of customers use paperless billing**.

### Target variable

The target variable `churn` is binary:

* `0`: customer did not leave.
* `1`: customer left.

The target distribution is:

| Churn | Customers | Percentage |
| ----- | --------: | ---------: |
| 0     |     5,174 |     73.46% |
| 1     |     1,869 |     26.54% |

This class distribution will be considered during the modeling stage. The class imbalance will be addressed **after the complete feature dataset has been prepared and before model training**, allowing the different models to be evaluated under comparable conditions.

### Main findings

The contract dataset provides several potentially useful predictors for churn, particularly:

* monthly charges;
* total charges;
* contract type;
* payment method;
* paperless billing;
* customer tenure, which can be derived from the available date information.

`end_date` will not be used directly as a predictor because it contains information about the customer's departure and was used to define the target variable itself. Likewise, `customer_id` will be retained only as an identifier for data integration and will not be used as a model feature.

The next stage will be to integrate the contract, personal, internet, and phone datasets, validate customer identifiers, create appropriate derived variables such as tenure, encode categorical variables, and prepare the final feature matrix before addressing class imbalance and comparing machine-learning models.

---

### Tenure
Tenure was calculated as the number of months between the customer's begin_date and a fixed reference date. The end_date variable was not used to calculate tenure because it represents the churn event and would introduce temporal leakage when predicting future customer churn.