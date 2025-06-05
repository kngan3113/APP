# recommend_app_upgraded.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load mô hình và dữ liệu
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
encoded_columns = joblib.load("encoded_columns.pkl")
recommendations = joblib.load("recommendations.pkl")

# Cấu hình giao diện
st.set_page_config(page_title="Gợi ý Dịch vụ", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🔍 GỢI Ý GÓI DỊCH VỤ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>AI phân tích thông tin khách hàng và đề xuất dịch vụ phù hợp nhất.</p>", unsafe_allow_html=True)

with st.form("customer_form"):
    st.markdown("""
    <div style='padding: 15px; background-color: #ffffff; border: 1px solid #ddd; border-radius: 10px;'>
    <h4 style='color:#6c63ff;'>🧾 Thông tin khách hàng</h4>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Giới tính", ["Male", "Female"])
    with c2:
        age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
    with c3:
        married = st.selectbox("Đã kết hôn", ["Yes", "No"])
    with c4:
        dependents = st.selectbox("Người phụ thuộc", ["Yes", "No"])

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        referrals = st.selectbox("Giới thiệu", ["Yes", "No"])
    with c6:
        tenure = st.slider("Thời gian sử dụng (tháng)", 0, 80, 12)
    with c7:
        monthly_charge = st.number_input("Cước hàng tháng ($)", value=70.0)
    with c8:
        total_charges = st.number_input("Tổng chi tiêu ($)", value=1000.0)

    st.markdown("<h5 style='color:#17a2b8;'>🌐 Dịch vụ chính</h5>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with d2:
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
    with d3:
        internet_service = st.selectbox("Internet Service", ["Yes", "No"])
    with d4:
        internet_type = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"])

    st.markdown("<h5 style='color:#ffc107;'>📦 Gói tiện ích</h5>", unsafe_allow_html=True)
    u1, u2, u3, u4 = st.columns(4)
    with u1:
        online_security = st.selectbox("Online Security", ["Yes", "No"])
    with u2:
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
    with u3:
        device_protection = st.selectbox("Device Protection Plan", ["Yes", "No"])
    with u4:
        tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])

    u5, u6, u7, u8 = st.columns(4)
    with u5:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
    with u6:
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
    with u7:
        streaming_music = st.selectbox("Streaming Music", ["Yes", "No"])
    with u8:
        unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No"])

    st.markdown("<h5 style='color:#007bff;'>💳 Hợp đồng & thanh toán</h5>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        contract = st.selectbox("Hợp đồng", ["Month-to-Month", "One Year", "Two Year"])
    with p2:
        paperless_billing = st.selectbox("Hóa đơn điện tử", ["Yes", "No"])
    with p3:
        payment_method = st.selectbox("Phương thức thanh toán", ["Bank Withdrawal", "Credit Card", "Mailed Check", "Electronic Check"])
    with p4:
        offer = st.selectbox("Khuyến mãi", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])

    submitted = st.form_submit_button("🚀 GỢI Ý NGAY")
    st.markdown("</div>", unsafe_allow_html=True)

if submitted:
    new_data = pd.DataFrame([{...}])  # giữ nguyên như bản trước bạn dùng
    encoded = pd.get_dummies(new_data).reindex(columns=encoded_columns, fill_value=0)
    scaled = scaler.transform(encoded)
    cluster = kmeans.predict(scaled)[0]
    recommendation = recommendations.get(cluster, "⚠️ Không tìm thấy gợi ý phù hợp.")

    st.markdown("---")
    st.markdown(f"### ✅ Khách hàng thuộc **CỤM {cluster}**")

    st.markdown(
        f"""<div style='background-color:#e6f4ea;padding:20px;border-left:5px solid #34a853;
        border-radius:6px;line-height:1.8;font-size:16px;white-space:pre-line;'>
        {recommendation}
        </div>""",
        unsafe_allow_html=True
    )

    st.download_button("📥 Tải gợi ý", recommendation, file_name=f"goi-y-cum-{cluster}.txt")
