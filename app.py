import streamlit as st
import pandas as pd
import pickle

# Load model
with open('random_forest_kidney_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data untuk preview
data = pd.read_csv('kidney_disease.csv')

st.title('Kidney Disease Prediction')

st.write('## Data Preview')
st.dataframe(data.head(20))

# Form input fitur (tanpa kolom id & classification)
input_columns = [col for col in data.columns if col not in ['id', 'classification']]
user_input = {}
for col in input_columns:
    if data[col].dtype == 'object' or data[col].isnull().any():
        user_input[col] = st.selectbox(col, data[col].dropna().unique())
    else:
        user_input[col] = st.number_input(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))

if st.button('Predict'):
    input_df = pd.DataFrame([user_input])
    # Encoding sederhana jika perlu (asumsi model butuh numerik)
    input_encoded = input_df.copy()
    for col in input_columns:
        if data[col].dtype == 'object' or data[col].isnull().any():
            unique_vals = list(data[col].dropna().unique())
            mapping = {val: i for i, val in enumerate(unique_vals)}
            input_encoded[col] = input_df[col].map(mapping)
    try:
        prediction = model.predict(input_encoded)[0]
        if prediction == 0:
            st.success('Hasil: Tidak terdeteksi penyakit ginjal')
            st.info('Saran: Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.')
        elif prediction == 1:
            st.warning('Hasil: Terdeteksi penyakit ginjal')
            st.info('Saran: Segera konsultasikan hasil ini ke dokter spesialis ginjal untuk penanganan lebih lanjut.')
        else:
            st.info(f'Hasil prediksi: {prediction}')
    except Exception as e:
        st.error(f'Error: {e}')
