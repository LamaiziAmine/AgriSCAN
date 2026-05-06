import streamlit as st
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import base64

# --- 1. CONFIGURATION ET CHARGEMENT ---
st.set_page_config(page_title="AgriScan", layout="wide")

@st.cache_resource
def load_my_model():
    # Charge ton modèle entraîné
    return tf.keras.models.load_model('model/plant_disease_model.h5',compile=False)

def load_data():
    base_path = os.path.dirname(__file__)
    json_path = os.path.join(base_path, 'recommendations.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# On charge les ressources
model = load_my_model()
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# --- 2. STYLE CSS (TON STYLE EXACT) ---
st.markdown(f"""
    <style>
    html {{ scroll-behavior: smooth; }}
            
    section.main > div {{ scroll-behavior: smooth; }}
    
    header {{visibility: hidden;}}
        
    #MainMenu {{visibility: hidden;}}
               
    footer {{visibility: hidden;}}
    .stApp {{
        background-image: linear-gradient(rgba(26, 28, 20, 0.85), rgba(26, 28, 20, 0.95)), 
                          url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        color: #F0F0F0;
    }}
    .nav-bar {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 0px 3%; background: rgba(0,0,0,0.5); backdrop-filter: blur(10px);
        position: fixed; top: 0; left: 0; width: 100%; z-index: 1000;
    }}
    .nav-btns {{ display: flex; gap: 20px; }}
    .btn-scan {{ background: #C5D145; color: #1A1C14 !important; padding: 8px 20px; border-radius: 5px; font-weight: bold; text-decoration : none !important ; }}
    .btn-login {{ border: 1px solid #C5D145; color: #C5D145; padding: 8px 20px; border-radius: 5px; }}
    .hero {{ text-align: center; padding-top: 80px; padding-bottom: 50px; }}
    .hero h1 {{ font-size: 60px; text-transform: uppercase; letter-spacing: 2px; }}
    .hero p {{ text-align: center; font-size: 20px; color: #BDC3C7; max-width: 700px; margin: auto; margin-top: 20px; }}
    .highlight {{ color: #C5D145; }}
    .description {{ font-size: 18px; color: #BDC3C7; max-width: 600px; margin: auto; }}
    .diag-card {{
        background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(197, 209, 69, 0.2);
        padding: 25px;margin-bottom : 80px ; border-radius: 15px; backdrop-filter: blur(5px);
    }}
    </style>
    """, unsafe_allow_html=True)


def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("app\logo.png")

st.components.v1.html("""
    <script>
    const links = window.parent.document.querySelectorAll('a[href="#scan-section"]');

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = window.parent.document.getElementById("scan-section");
            if (target) {
                target.scrollIntoView({ behavior: "smooth" });
            }
        });
    });
    </script>
""", height=0)

# --- HEADER / NAVBAR ---
st.markdown(f"""
    <div class="nav-bar">
        <div style="display: flex; align-items: center;">
          <img src="data:image/png;base64,{img_base64}" style="height: 90px; margin-right: 5px;">
          <div style="font-size: 34px; font-weight: bold;">Agri<span class="highlight">SCAN</span></div>
        </div>
        <div class="nav-btns">
            <a href="#scan-section" class="btn-scan">
            SCAN
            </a>
            <div class="btn-login">SE CONNECTER</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 4. HERO SECTION ---
st.markdown("""
    <div class="hero">
        <h1>Découvrez une Expérience <span class="highlight">Unique</span></h1>
        <p class="description">Protégez vos récoltes grâce à notre IA de précision. Identifiez les maladies instantanément et recevez des conseils d'experts basés sur vos conditions réelles.</p>
    </div>
    """, unsafe_allow_html=True)

# --- 5. ZONE DE DIAGNOSTIC ---
st.markdown('<div id="scan-section" class="diag-card">', unsafe_allow_html=True)
col_setup, col_result = st.columns([1, 1], gap="large")

with col_setup:
    st.subheader("1. Configuration")
    culture = st.selectbox("Choisir votre culture", ["Tomate", "Pomme de Terre", "Poivron"])
    temp = st.slider("Température actuelle (°C)", 0, 50, 25)
    hum = st.slider("Humidité (%)", 0, 100, 60)
    img_file = st.file_uploader("Prendre une photo claire de la feuille", type=["jpg", "png"])

with col_result:
    st.subheader("2. Résultats de l'IA")
    if img_file:
        image = Image.open(img_file)
        st.image(image, width=300)
        
        if st.button("Lancer l'analyse"):
            # --- TRAITEMENT IMAGE ---
            img_resized = image.resize((224, 224))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, 0) # Format [1, 224, 224, 3]
            
            # --- PRÉDICTION ---
            predictions = model.predict(img_array)
            index = np.argmax(predictions[0])
            prediction_label = CLASS_NAMES[index]
            confidence = np.max(predictions[0]) * 100
            
            # --- RÉCUPÉRATION JSON ---
            rec_json = load_data()
            data = rec_json.get(prediction_label, {})
            
            if data:
                st.markdown(f"### Maladie : <span class='highlight'>{data['title']}</span>", unsafe_allow_html=True)
                st.write(f"Indice de confiance IA : **{confidence:.2f}%**")
                
                # --- LOGIQUE DE RISQUE ET CONSEILS RÉELS ---
                st.write("---")
                factors = data.get('risk_factors', {})
                ideal = data.get('ideal_plant_conditions', {})
                
                risk_score = 0
                alerts = []
                
                if temp >= factors['temp_min'] and temp <= factors['temp_max']:
                    risk_score += 50
                    alerts.append(f"⚠️ Température critique pour cette maladie. Action : Visez **{ideal['temp_target']}°C**.")
                
                if hum >= factors['hum_min']:
                    risk_score += 50
                    alerts.append(f"🚨 Humidité trop élevée. Action : Réduisez à **{ideal['hum_target']}%**.")
                
                # Affichage du score de danger
                color = "red" if risk_score >= 100 else "orange" if risk_score >= 50 else "green"
                st.markdown(f"**Niveau de Danger de propagation :** <span style='color:{color}; font-weight:bold;'>{risk_score}%</span>", unsafe_allow_html=True)
                
                # Affichage des alertes climatiques
                for a in alerts:
                    st.warning(a)
                
                # Actions pratiques du JSON
                st.write("**✅ Actions recommandées :**")
                for action in data.get('practical_actions', []):
                    st.write(f"- {action}")
                
                st.info(f"💡 **Traitement Bio :** {data.get('treatment_organic', 'N/A')}")
            else:
                st.error("Erreur : Maladie non trouvée dans la base de données.")

st.markdown('</div>', unsafe_allow_html=True)