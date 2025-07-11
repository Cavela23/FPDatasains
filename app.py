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
# Keterangan untuk field tertentu
field_help = {
    'bp': 'Blood pressure (tekanan darah)',
    'sg': 'Specific gravity (berat jenis urin)',
    'al': 'Albumin (kadar albumin urin)',
    'su': 'Sugar (kadar gula urin)',
    'rbc': 'Red blood cells (sel darah merah)',
    'pc': 'Pus cell (sel nanah)',
    'pcc': 'Pus cell clumps (gumpalan sel nanah)',
    'ba': 'Bacteria (bakteri)',
    'bgr': 'Blood glucose random (gula darah acak)',
    'bu': 'Blood urea (urea darah)',
    'sc': 'Serum creatinine (kreatinin serum)',
    'sod': 'Sodium (natrium)',
    'pot': 'Potassium (kalium)',
    'hemo': 'Hemoglobin',
    'pcv': 'Packed cell volume (volume sel darah merah)',
    'wc': 'White blood cell count (jumlah sel darah putih)',
    'rc': 'Red blood cell count (jumlah sel darah merah)',
    'htn': 'Hypertension (tekanan darah tinggi)',
    'dm': 'Diabetes mellitus',
    'cad': 'Coronary artery disease (penyakit jantung koroner)',
    'appet': 'Appetite (nafsu makan)',
    'pe': 'Edema/pembengkakan',
    'ane': 'Anemia',
}
for col in input_columns:
    help_text = field_help.get(col, None)
    label = f"{col} - {help_text}" if help_text else col
    # Coba konversi ke float, jika bisa berarti numerik
    try:
        data[col] = pd.to_numeric(data[col])
        val = st.text_input(label, value=str(data[col].mean()))
        try:
            user_input[col] = float(val)
        except ValueError:
            user_input[col] = None
    except Exception:
        user_input[col] = st.selectbox(label, data[col].dropna().unique())

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
