import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import os

# Ensure TensorFlow is using GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and enabled.")
else:
    print("No GPU found. Running on CPU.")

# Define dataset directory
dataset_dir = "dataset"  # Adjust as necessary
img_size = (224, 224)
batch_size = 32

# Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load Pre-trained Models (ResNet50 & VGG16)
def create_model(base_model):
    base_model.trainable = False
    global_avg_pool = GlobalAveragePooling2D()(base_model.output)
    output_layer = Dense(len(train_generator.class_indices), activation='softmax')(global_avg_pool)
    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

resnet_model = create_model(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
vgg_model = create_model(VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))

#%%

# Train ResNet50 Model
resnet_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1
)
resnet_model.save("apple_classifier_resnet50.h5")

# Train VGG16 Model
vgg_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1
)
vgg_model.save("apple_classifier_vgg16.h5")

print("Models training complete and saved.")

#%%

# Feature Extraction using SURF or ORB
def extract_features(img_dir, use_surf=True):
    feature_extractor = cv2.xfeatures2d.SURF_create() if use_surf else cv2.ORB_create()
    features, labels = [], []
    class_labels = os.listdir(img_dir)
    for label in class_labels:
        class_path = os.path.join(img_dir, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            keypoints, descriptors = feature_extractor.detectAndCompute(img, None)
            if descriptors is not None:
                features.append(descriptors.flatten())
                labels.append(label)
    return np.array(features, dtype=object), np.array(labels)

X, y = extract_features(dataset_dir, use_surf=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply ML models
models = {
    "SVM": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

for name, clf in models.items():
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"{name} Accuracy: {scores.mean():.4f}")
