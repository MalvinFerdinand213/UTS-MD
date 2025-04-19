import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Misal model kamu adalah random_forest
joblib.dump(random_forest, 'model.pkl')

# Load model
model = joblib.load('model.pkl')

# Judul halaman
st.title("Prediksi Pembatalan Booking Hotel")

# Input dari user
hotel_type = st.selectbox("Tipe Hotel", ["City Hotel", "Resort Hotel"])
lead_time = st.number_input("Lead Time (hari)", min_value=0)
adults = st.number_input("Jumlah Dewasa", min_value=1)
children = st.number_input("Jumlah Anak", min_value=0)
weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0)
week_nights = st.number_input("Malam Hari Kerja", min_value=0)
previous_cancellations = st.number_input("Pembatalan Sebelumnya", min_value=0)

# Preprocessing manual jika perlu (contoh encoding)
hotel_encoded = 1 if hotel_type == "City Hotel" else 0

# Gabung input dalam array untuk prediksi
features = np.array([[hotel_encoded, lead_time, adults, children,
                      weekend_nights, week_nights, previous_cancellations]])

# Tombol prediksi
if st.button("Prediksi"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Booking kemungkinan akan DIBATALKAN")
    else:
        st.success("Booking kemungkinan TIDAK DIBATALKAN")
