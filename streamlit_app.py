import streamlit as st
import joblib
import pandas as pd

# Load model pipeline (model dan preprocessing dalam satu kesatuan)
pipeline = joblib.load('trained_model.pkl')

st.title("Prediksi Pembatalan Booking Hotel")

# Input dari user
no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, value=2)
no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Jumlah Malam Akhir Pekan", min_value=0)
no_of_week_nights = st.number_input("Jumlah Malam Hari Kerja", min_value=0)
required_car_parking_space = st.selectbox("Perlu Parkir Mobil?", [0, 1])
lead_time = st.number_input("Lead Time (hari sebelum check-in)", min_value=0)
arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
arrival_month = st.selectbox("Bulan Kedatangan", list(range(1, 13)))
arrival_date = st.selectbox("Tanggal Kedatangan", list(range(1, 32)))
repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0)
no_of_previous_bookings_not_canceled = st.number_input("Jumlah Booking Lalu Tidak Dibatalkan", min_value=0)
avg_price_per_room = st.number_input("Harga Rata-rata per Kamar", min_value=0.0)
no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0)

# Susun input sebagai DataFrame agar sesuai dengan pipeline
user_input = pd.DataFrame({
    'no_of_adults': [no_of_adults],
    'no_of_children': [no_of_children],
    'no_of_weekend_nights': [no_of_weekend_nights],
    'no_of_week_nights': [no_of_week_nights],
    'required_car_parking_space': [required_car_parking_space],
    'lead_time': [lead_time],
    'arrival_year': [arrival_year],
    'arrival_month': [arrival_month],
    'arrival_date': [arrival_date],
    'repeated_guest': [repeated_guest],
    'no_of_previous_cancellations': [no_of_previous_cancellations],
    'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
    'avg_price_per_room': [avg_price_per_room],
    'no_of_special_requests': [no_of_special_requests]
})

# Tampilkan data input untuk debugging
st.write("Data Input yang Dikirimkan ke Model:")
st.write(user_input)

# Prediksi
if st.button("Prediksi"):
    try:
        # Memastikan prediksi dilakukan sesuai pipeline
        prediction = pipeline.predict(user_input)  # Menggunakan pipeline untuk memproses input dan prediksi
        if prediction[0] == 1:
            st.error("Booking kemungkinan DIBATALKAN")
        else:
            st.success("Booking kemungkinan TIDAK DIBATALKAN")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
