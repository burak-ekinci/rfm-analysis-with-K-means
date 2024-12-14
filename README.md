# RFM Analysis Project

This project uses the **RFM (Recency, Frequency, Monetary)** analysis technique to perform customer segmentation. It identifies customer behavior patterns and provides actionable insights to develop marketing strategies.

---

## Project Structure

The project follows the **Data Analytics Lifecycle**:

### 1. **Problem Definition**

- **Objective**: Group customers based on their Recency, Frequency, and Monetary values to understand their behavior and improve engagement.
- **Business Use Cases**:
  - Customer segmentation
  - Targeted marketing campaigns
  - Optimized sales and promotion strategies

---

### 2. **Data Collection**

- **Dataset**: Online retail transaction data (e.g., Kaggle datasets).
- **Key Features**:
  - `CustomerID`: Unique customer identifier
  - `InvoiceDate`: Date of transaction
  - `Quantity`: Number of items purchased
  - `UnitPrice`: Price per item

---

### 3. **Data Preprocessing**

Cleaning and transforming the raw data into a usable format:

- **Handling Missing Values**:

  ```python
  data = data.dropna()  # Drop rows with missing values
  ```

- **Removing Negative Values**:

  ```python
  data = data[data["Total"] > 0]  # Remove rows with invalid totals
  ```

- **Feature Creation**:
  - Calculate `Total` spend per transaction:
    ```python
    data["Total"] = data["Quantity"] * data["UnitPrice"]
    ```
  - Convert `InvoiceDate` to datetime:
    ```python
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    ```

---

### 4. **Exploratory Data Analysis (EDA)**

Performed analysis to understand the data:

- Total spending distribution
- Purchase frequency by customer
- Recency distribution (days since last purchase)

---

### 5. **RFM Metric Calculation**

The RFM metrics were calculated for each customer:

- **Recency**: Number of days since the last purchase
- **Frequency**: Total number of transactions
- **Monetary**: Total spending by the customer

```python
rfm = data.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (data["InvoiceDate"].max() - x.max()).days,
    "InvoiceNo": "count",
    "Total": "sum"
})
rfm.columns = ["Recency", "Frequency", "Monetary"]
```

---

### 6. **Clustering (K-Means)**

The K-Means clustering algorithm was used to segment customers into groups:

1. **Standardize Data**:

   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   rfm_scaled = scaler.fit_transform(rfm)
   ```

2. **Train K-Means Model**:
   ```python
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=4, random_state=42)
   rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
   ```

---

### 7. **Visualization of Results**

The clusters were analyzed and visualized to derive insights:

- Average Recency, Frequency, and Monetary values by cluster
- Distribution plots of RFM metrics

Example:

```python
import seaborn as sns
sns.barplot(x="Cluster", y="Recency", data=rfm)
```

---

### 8. **Business Insights and Actions**

The clusters were interpreted, and actionable marketing strategies were developed:

| **Cluster** | **Recency** | **Frequency** | **Monetary** | **Business Insight**           | **Action Plan**                 |
| ----------- | ----------- | ------------- | ------------ | ------------------------------ | ------------------------------- |
| **0**       | Low         | High          | High         | Loyal and high-value customers | Personalized offers, rewards    |
| **1**       | Medium      | Low           | Medium       | Dormant customers              | Reminder campaigns              |
| **2**       | High        | Low           | Low          | Lost customers                 | Re-engagement campaigns         |
| **3**       | Medium      | Medium        | Medium       | Potential loyal customers      | Incentives and loyalty programs |

---

## Project Setup

### Prerequisites

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Run the Project

Execute the following script to run the analysis:

```bash
python rfm_analysis.py
```

---

## Libraries Used

- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
