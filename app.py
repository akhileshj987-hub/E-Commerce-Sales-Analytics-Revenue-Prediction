import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="E-Commerce Analytics", page_icon="🛒", layout="wide")

DATA_FILE = "ecommerce_sales_dataset.csv"
MODEL_FILE = "random_forest_revenue_model.pkl"


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    required = [
        "Order_ID", "Order_Date", "Customer_ID", "City", "Category",
        "Product", "Quantity", "Unit_Price", "Discount",
        "Payment_Method", "Order_Status",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(missing))

    data = data.drop_duplicates().copy()
    data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")

    for c in ["Quantity", "Unit_Price", "Discount"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    for c in ["City", "Payment_Method", "Category", "Product", "Order_Status"]:
        if data[c].isna().any():
            modes = data[c].mode(dropna=True)
            data[c] = data[c].fillna(modes.iloc[0] if not modes.empty else "Unknown")

    if data["Discount"].isna().any():
        median_discount = data["Discount"].median()
        data["Discount"] = data["Discount"].fillna(
            median_discount if pd.notna(median_discount) else 0
        )

    data = data.dropna(subset=["Order_Date", "Quantity", "Unit_Price"]).copy()
    data = data[(data["Quantity"] >= 0) & (data["Unit_Price"] >= 0)].copy()

    # Convert whole-number discounts (e.g. 10) to fractions (0.10).
    if not data.empty and data["Discount"].max() > 1:
        data["Discount"] = data["Discount"] / 100
    data["Discount"] = data["Discount"].clip(0, 1)

    data["Revenue"] = data["Quantity"] * data["Unit_Price"] * (1 - data["Discount"])
    data["Month"] = data["Order_Date"].dt.to_period("M").dt.to_timestamp()
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"Cannot find {DATA_FILE}. Put it in the same folder as app.py, "
        "then run Streamlit from that folder."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if df.empty:
    st.error("The dataset has no usable rows after cleaning.")
    st.stop()

st.sidebar.title("🛒 E-Commerce Analytics")
st.sidebar.caption("Sales insights and revenue estimation")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Dashboard", "Revenue Predictor", "Dataset Explorer"],
)

if page == "Sales Dashboard":
    st.title("📊 E-Commerce Sales Dashboard")
    st.caption("Explore revenue, order activity, and sales trends.")

    total_revenue = float(df["Revenue"].sum())
    total_orders = int(df["Order_ID"].nunique())
    total_units = float(df["Quantity"].sum())
    average_order_value = total_revenue / total_orders if total_orders else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
    c2.metric("🛒 Unique Orders", f"{total_orders:,}")
    c3.metric("📦 Units Sold", f"{total_units:,.0f}")
    c4.metric("🧾 Avg. Revenue per Order", f"₹{average_order_value:,.2f}")

    st.divider()
    st.subheader("🔎 Filter dashboard")
    min_date = df["Order_Date"].min().date()
    max_date = df["Order_Date"].max().date()
    d1, d2 = st.columns(2)
    start_date = d1.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = d2.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.warning("Start date is after end date. Showing all dates instead.")
        filtered_df = df.copy()
    else:
        filtered_df = df[
            (df["Order_Date"].dt.date >= start_date)
            & (df["Order_Date"].dt.date <= end_date)
        ].copy()

    if filtered_df.empty:
        st.warning("No records match this date range. Select a wider range.")
    else:
        revenue = float(filtered_df["Revenue"].sum())
        orders = int(filtered_df["Order_ID"].nunique())
        units = float(filtered_df["Quantity"].sum())
        aov = revenue / orders if orders else 0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Revenue in Range", f"₹{revenue:,.0f}")
        m2.metric("Orders in Range", f"{orders:,}")
        m3.metric("Units in Range", f"{units:,.0f}")
        m4.metric("Revenue / Order", f"₹{aov:,.2f}")

        st.divider()
        st.subheader("🏆 Revenue by Product Category")
        category_revenue = (
            filtered_df.groupby("Category", as_index=False)["Revenue"]
            .sum().sort_values("Revenue", ascending=True)
        )
        if not category_revenue.empty:
            category_revenue["Revenue Share (%)"] = (
                category_revenue["Revenue"] / category_revenue["Revenue"].sum() * 100
            )
            fig = px.bar(
                category_revenue, x="Revenue", y="Category", orientation="h",
                color="Category", text="Revenue",
                hover_data={"Revenue": ":,.2f", "Revenue Share (%)": ":.2f"},
                labels={"Revenue": "Revenue (₹)", "Category": "Product category"},
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(texttemplate="₹%{x:,.0f}", textposition="outside", cliponaxis=False)
            fig.update_layout(
                height=430, showlegend=False, xaxis_title="Revenue (₹)", yaxis_title="",
                margin=dict(l=10, r=90, t=25, b=20), hoverlabel=dict(font_size=13),
            )
            st.plotly_chart(fig, use_container_width=True, key="category_revenue_chart")

        st.divider()
        st.subheader("📈 Monthly Revenue Trend")
        monthly = (
            filtered_df.groupby("Month", as_index=False)["Revenue"]
            .sum().sort_values("Month")
        )
        if not monthly.empty:
            fig = px.area(
                monthly, x="Month", y="Revenue", markers=True,
                labels={"Month": "Month", "Revenue": "Revenue (₹)"},
                template="plotly_dark",
            )
            fig.update_traces(
                line=dict(width=3), marker=dict(size=8),
                hovertemplate="<b>%{x|%B %Y}</b><br>Revenue: ₹%{y:,.2f}<extra></extra>",
            )
            peak = monthly.loc[monthly["Revenue"].idxmax()]
            fig.add_annotation(
                x=peak["Month"], y=peak["Revenue"], text="Highest month",
                showarrow=True, arrowhead=2, ax=0, ay=-40,
            )
            fig.update_layout(
                height=430, xaxis_title="Month", yaxis_title="Revenue (₹)",
                hovermode="x unified", margin=dict(l=10, r=20, t=55, b=20),
            )
            st.plotly_chart(fig, use_container_width=True, key="monthly_revenue_chart")

        st.divider()
        st.subheader("🍩 Order Status Distribution")
        status_data = (
            filtered_df["Order_Status"].fillna("Unknown").value_counts()
            .rename_axis("Order Status").reset_index(name="Record Count")
        )
        if not status_data.empty:
            fig = px.pie(
                status_data, names="Order Status", values="Record Count",
                hole=0.62, template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(
                textinfo="percent+label", textposition="inside",
                hovertemplate="<b>%{label}</b><br>Records: %{value:,}<br>Share: %{percent}<extra></extra>",
                marker=dict(line=dict(color="#111827", width=2)),
            )
            fig.update_layout(
                height=430, showlegend=True, legend_title_text="Order status",
                margin=dict(l=10, r=10, t=25, b=20),
                annotations=[dict(
                    text=f"{int(status_data['Record Count'].sum()):,}<br>Records",
                    x=0.5, y=0.5, font_size=20, showarrow=False,
                )],
            )
            st.plotly_chart(fig, use_container_width=True, key="order_status_chart")

        st.caption(
            "Status distribution counts dataset rows; the KPI counts unique Order_IDs. "
            "Revenue is Quantity × Unit Price × (1 − Discount)."
        )

elif page == "Revenue Predictor":
    st.title("🛒 E-Commerce Revenue Predictor")
    st.caption("Estimate order revenue with your saved Random Forest model.")

    quantity = st.number_input("Quantity", min_value=1, max_value=1000, value=3, step=1)
    unit_price = st.number_input(
        "Unit Price (₹)", min_value=0.01, max_value=1_000_000.0,
        value=2000.0, step=100.0,
    )
    discount_percent = st.slider("Discount (%)", min_value=0, max_value=90, value=10)

    if st.button("Predict Revenue", type="primary"):
        discount = discount_percent / 100.0
        new_order = pd.DataFrame({
            "Quantity": [quantity],
            "Unit_Price": [unit_price],
            "Discount": [discount],
        })
        try:
            model = load_model()
            prediction = float(model.predict(new_order)[0])
            formula_revenue = quantity * unit_price * (1 - discount)
            r1, r2 = st.columns(2)
            r1.metric("🤖 ML Predicted Revenue", f"₹{prediction:,.2f}")
            r2.metric("🧮 Formula-based Revenue", f"₹{formula_revenue:,.2f}")
            st.info(
                "This is a learning project using synthetic practice data. Revenue is directly "
                "determined by the input formula, so this model learns that formula rather "
                "than forecasting real-world sales."
            )
        except FileNotFoundError:
            st.error(f"Cannot find {MODEL_FILE}. Put it in the same folder as app.py.")
        except Exception as error:
            st.error(
                f"Could not load or use the model: {error}. If the model was saved with a "
                "different scikit-learn version, retrain/save it in this Python environment."
            )

elif page == "Dataset Explorer":
    st.title("🔍 Dataset Explorer")
    st.caption("Inspect the cleaned dataset and download a copy.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Duplicate rows remaining", f"{int(df.duplicated().sum()):,}")

    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.subheader("Download cleaned dataset")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_ecommerce_sales.csv",
        mime="text/csv",
    )
