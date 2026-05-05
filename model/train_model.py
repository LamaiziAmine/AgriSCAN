import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_SET_PATH = "data/"  # Dossier où se trouvent tes dossiers de classes
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 # Tu peux augmenter à 20 ou 25 si tu as un bon PC

# ==========================================
# 2. CHARGEMENT DES DONNÉES (PRÉTRAITEMENT 1)
# ==========================================
print("Chargement des données...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_SET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int' 
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_SET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes trouvées : {class_names}")

# Optimisation pour la rapidité
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. CRÉATION DU MODÈLE AVEC AUGMENTATION
# ==========================================

# Bloc de Data Augmentation (Prétraitement 2)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Import du modèle pré-entraîné MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False, 
    weights='imagenet'
)
base_model.trainable = False # On ne touche pas aux poids de base de Google

# Assemblage du modèle final
model = models.Sequential([
    # Entrée
    layers.Input(shape=(224, 224, 3)),
    
    # Étape 1 : Augmentation (uniquement active pendant le training)
    data_augmentation,
    
    # Étape 2 : Normalisation (Pixels 0-255 -> -1 à 1)
    layers.Rescaling(1./127.5, offset=-1),
    
    # Étape 3 : Le "corps" du CNN (MobileNetV2)
    base_model,
    
    # Étape 4 : La "tête" de classification (ton intelligence spécifique)
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # Évite le surapprentissage
    layers.Dense(num_classes, activation='softmax') # Sortie : 1 probabilité par maladie
])

# ==========================================
# 4. COMPILATION ET ENTRAÎNEMENT
# ==========================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\nDébut de l'entraînement...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================================
# 5. SAUVEGARDE ET VISUALISATION
# ==========================================
# Sauvegarder le modèle
model.save('model/plant_disease_model.h5')
print("\nModèle sauvegardé avec succès dans le dossier 'model/' !")

# Afficher les courbes de performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Précision Entraînement')
plt.plot(val_acc, label='Précision Validation')
plt.legend()
plt.title('Précision')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Perte Entraînement')
plt.plot(val_loss, label='Perte Validation')
plt.legend()
plt.title('Perte (Loss)')
plt.show()