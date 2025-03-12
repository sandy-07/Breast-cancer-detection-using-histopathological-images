import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import InceptionV3 # type: ignore
import os

# Define Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "dataset"  # Ensure this contains 'train' & 'validation' folders

# Check if dataset exists
if not os.path.exists(f"{DATASET_DIR}/train") or not os.path.exists(f"{DATASET_DIR}/validation"):
    raise FileNotFoundError("⚠️ Dataset directories not found! Ensure 'dataset/train' and 'dataset/validation' exist.")

# Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load Train Data
train_data = train_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Load Validation Data
val_data = val_datagen.flow_from_directory(
    f"{DATASET_DIR}/validation",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

# Load Pretrained Model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze layers

# Build Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary Classification
])

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("breast_cancer_model.keras", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
]

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

# Save the trained model
model.save("breast_cancer_model.keras")
model.save("breast_cancer_model.h5")  # Optional

print("✅ Model training complete and saved as 'breast_cancer_model.keras' & 'breast_cancer_model.h5'")
