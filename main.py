import tensorflow as tf
import numpy as np
import os

# ==========================================
# 1. LISTE EXACTE DES CLASSES (COPIÉE DE TES LOGS)
# ==========================================
# C'est cette liste qui garantit que l'index 0 correspond bien à Pepper Bacterial Spot
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

MODEL_PATH = 'model/plant_disease_model.h5'

# Chargement du modèle
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--- Modele charge avec succes ---")
else:
    print(f"ERREUR : Fichier {MODEL_PATH} introuvable.")

def predict_disease(image_path):
    # SOLUTION 2 : Utiliser les outils Keras pour charger l'image
    # Cela garantit le même redimensionnement et le mode RGB que l'entraînement
    try:
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Créer le batch (1, 224, 224, 3)

        # Prédiction
        predictions = model.predict(img_array)
        
        # Récupération des résultats
        index = np.argmax(predictions[0])
        label = CLASS_NAMES[index]
        confidence = 100 * predictions[0][index]
        
        return label, confidence
    except Exception as e:
        return f"Erreur lors du traitement : {e}", 0

if __name__ == "__main__":
    # Remplace par le nom de ton image de test
    PATH_TO_IMAGE = "images1.jpg" 
    
    if os.path.exists(PATH_TO_IMAGE):
        disease, conf = predict_disease(PATH_TO_IMAGE)
        print("-" * 30)
        print(f"RÉSULTAT : {disease}")
        print(f"CONFIANCE : {conf:.2f}%")
        print("-" * 30)
    else:
        print(f"Fichier {PATH_TO_IMAGE} introuvable dans le dossier.")