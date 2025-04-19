import streamlit as st
import joblib
import numpy as np
import pandas as pd

user_input = [erythema, scaling, definite_borders, itching, koebner_phenomenon, polygonal_papules, follicular_papules, oral_mucosal_involvement,
                knee_and_elbow_involvement, scalp_involvement, family_history, melanin_incontinence, eosinophils_infiltrate, PNL_infiltrate, 
                fibrosis_papillary_dermis, exocytosis, acanthosis, hyperkeratosis, parakeratosis, clubbing_rete_ridges, elongation_rete_ridges,\
                thinning_suprapapillary_epidermis, spongiform_pustule, munro_microabcess, focal_hypergranulosis, disappearance_granular_layer,
                vacuolisation_damage_basal_layer, spongiosis, saw_tooth_appearance_retes, follicular_horn_plug, perifollicular_parakeratosis,\
                inflammatory_mononuclear_infiltrate, band_like_infiltrate, age]

  model_filename = 'trained_model.pkl'
  model = load_model(model_filename)
  prediction = predict_with_model(model, user_input)
  st.write('The prediction output is: ', prediction)
features = np.array([[hotel_encoded, lead_time, adults, children,
                      weekend_nights, week_nights, previous_cancellations]])

# Tombol prediksi
if st.button("Prediksi"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Booking kemungkinan akan DIBATALKAN")
    else:
        st.success("Booking kemungkinan TIDAK DIBATALKAN")
