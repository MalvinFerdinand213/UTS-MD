import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load model
model = pd.read_csv("Dataset_B_hotel.csv") 

st.set_page_config(page_title="Hotel Booking Prediction", layout="centered")
st.title("🛎️ Prediksi Status Booking Hotel")

# Input form
with st.form("input_form"):
    st.subheader("Masukkan Data Booking")
    no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, value=2)
    no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Jumlah Malam Akhir Pekan", min_value=0, value=1)
    no_of_week_nights = st.number_input("Jumlah Malam Hari Kerja", min_value=0, value=2)
    type_of_meal_plan = st.selectbox("Paket Makan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
    required_car_parking_space = st.radio("Butuh Parkir?", [0, 1])
    room_type_reserved = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4"])
    lead_time = st.number_input("Lead Time (hari sebelum kedatangan)", min_value=0, value=30)
    arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
    arrival_month = st.slider("Bulan Kedatangan", 1, 12, 6)
    arrival_date = st.slider("Tanggal Kedatangan", 1, 31, 15)
    market_segment_type = st.selectbox("Segmen Pasar", ["Online", "Offline", "Corporate", "Aviation"])
    repeated_guest = st.radio("Tamu Berulang?", [0, 1])
    no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Booking Sebelumnya yang Tidak Dibatalkan", min_value=0, value=0)
    avg_price_per_room = st.number_input("Harga Rata-rata Kamar", min_value=0.0, value=100.0)
    no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, value=0)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_dict = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }

    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    
    if prediction == "Canceled":
        st.error(f"❌ Booking kemungkinan **DIBATALKAN**.")
    else:
        st.success(f"✅ Booking kemungkinan **TIDAK DIBATALKAN**.")
