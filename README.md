# 🛒 E-Commerce Sales Analytics & Revenue Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

<p align="center">
  An interactive Streamlit dashboard for exploring e-commerce sales performance and estimating order revenue.
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-how-it-works">How it works</a> •
  <a href="#-project-structure">Project structure</a>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Sales Dashboard | View total revenue, unique orders, units sold, and average revenue per order. |
| 📅 Date Filters | Explore metrics and charts for a selected date range. |
| 📈 Interactive Charts | Analyze category revenue, monthly revenue trends, and order-status distribution. |
| 🤖 Revenue Predictor | Estimate revenue using a trained Random Forest regression model. |
| 🔍 Dataset Explorer | Preview cleaned data, inspect descriptive statistics, and download the processed CSV. |
| 🧹 Data Cleaning | Removes duplicates, handles missing values, validates numeric fields, and normalizes discounts. |

---

## 🖥️ App Pages

<details>
<summary><b>📊 Sales Dashboard</b></summary>

- Total revenue, order count, quantity sold, and average order value
- Revenue by product category
- Monthly revenue trend
- Order-status distribution
- Date-range filtering

</details>

<details>
<summary><b>🤖 Revenue Predictor</b></summary>

Enter:

- Quantity
- Unit price
- Discount percentage

The app returns:

- ML-predicted revenue
- Formula-based revenue

Revenue formula:

```text
Revenue = Quantity × Unit Price × (1 − Discount)
