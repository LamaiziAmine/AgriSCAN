import streamlit as st
import json
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuration
st.set_page_config(page_title="AgriScan", layout="wide")

# --- CHARGEMENT DES DONNÉES ---
def load_data():
    with open('recommendations.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# --- STYLE CSS (Le look "Unique Experience") ---
st.markdown(f"""
    <style>
    /* Global */
    .stApp {{
        background-image: linear-gradient(rgba(26, 28, 20, 0.85), rgba(26, 28, 20, 0.95)), 
                          url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        color: #F0F0F0;
    }}

    /* Navbar */
    .nav-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 5%;
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        position: fixed;
        top: 0; left: 0; width: 100%; z-index: 1000;
    }}
    .nav-btns {{ display: flex; gap: 20px; }}
    .btn-scan {{ background: #C5D145; color: #1A1C14; padding: 8px 20px; border-radius: 5px; font-weight: bold; cursor: pointer; }}
    .btn-login {{ border: 1px solid #C5D145; color: #C5D145; padding: 8px 20px; border-radius: 5px; cursor: pointer; }}

    /* Hero Section */
    .hero {{ text-align: center; padding-top: 150px; padding-bottom: 50px; }}
    .hero h1 {{ font-size: 60px; text-transform: uppercase; letter-spacing: 2px; }}
    .highlight {{ color: #C5D145; }}
    .description {{ font-size: 18px; color: #BDC3C7; max-width: 600px; margin: auto; }}

    /* Cards */
    .diag-card {{
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(197, 209, 69, 0.2);
        padding: 25px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }}
    </style>
    """, unsafe_allow_html=True)

# --- HEADER / NAVBAR ---
st.markdown(f"""
    <div class="nav-bar">
        <div style="font-size: 24px; font-weight: bold;">[ LOGO PLACEHOLDER ]</div>
        <div class="nav-btns">
            <div class="btn-scan">SCAN</div>
            <div class="btn-login">SE CONNECTER</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
    <div class="hero">
        <h1>Découvrez une Expérience <span class="highlight">Unique</span></h1>
        <p class="description">Protégez vos récoltes grâce à notre IA de précision. Identifiez les maladies instantanément et recevez des conseils d'experts basés sur vos conditions réelles.</p>
    </div>
    """, unsafe_allow_html=True)

# --- ZONE DE DIAGNOSTIC ---
st.markdown('<div class="diag-card">', unsafe_allow_html=True)
col_setup, col_result = st.columns([1, 1], gap="large")

with col_setup:
    st.subheader("1. Configuration")
    culture = st.selectbox("Choisir votre culture", ["Tomate", "Pomme de Terre", "Poivron"])
    temp = st.slider("Température actuelle (°C)", 0, 50, 25)
    hum = st.slider("Humidité (%)", 0, 100, 60)
    img_file = st.file_uploader("Prendre une photo de la feuille", type=["jpg", "png"])

with col_result:
    st.subheader("2. Résultats de l'IA")
    if img_file:
        st.image(img_file, width=300)
        if st.button("Lancer l'analyse"):
            # Simulation prédiction (remplace par ton model.predict)
            prediction = "Tomato_Late_blight" 
            
            # Récupération depuis le JSON
            data = load_data().get(prediction, {})
            
            st.markdown(f"### Maladie : <span class='highlight'>{data['title']}</span>", unsafe_allow_html=True)
            
            # --- Calcul intelligent du Risque ---
            factors = data.get('risk_factors', {})
            risk_score = 0
            if temp >= factors['temp_min'] and temp <= factors['temp_max']: risk_score += 50
            if hum >= factors['hum_min']: risk_score += 50
            
            color = "red" if risk_score > 70 else "orange" if risk_score > 30 else "green"
            st.markdown(f"**Niveau de Danger :** <span style='color:{color}'>{risk_score}%</span>", unsafe_allow_html=True)
            
            st.write(f"**Conseil :** {data['treatment_organic']}")

st.markdown('</div>', unsafe_allow_html=True)