import scipy.io
import numpy as np

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the MATLAB file
mat_file = 'umist_cropped.mat'
data = scipy.io.loadmat(mat_file)

# Print variable names and shapes
for key in data:
    if not key.startswith('__'):
        print(f"Variable: {key}, Shape: {data[key].shape}")

# Extract images and labels
images = data['facedat'][0]  # List of image arrays
labels = np.repeat(np.arange(len(data['dirnames'][0])), [len(item) for item in data['facedat'][0]])

# Flatten the images and ensure they are of the same shape
images_flattened = []
for sublist in images:
    for img in sublist:
        if img.shape != (112, 92): 
            img = np.resize(img, (112, 92))
        images_flattened.append(img.flatten())
images_flattened[0]

# Normalize the images
images_normalized = np.array(images_flattened) / 255.0

print(f"Images shape: {images_normalized.shape}, Each image shape: {images_normalized[0].shape if images_normalized.size > 0 else 'N/A'}")
print(f"Labels shape: {labels.shape}")

# Splitting the Dataset
from sklearn.model_selection import train_test_split

# Stratified split
X_train, X_temp, y_train, y_temp = train_test_split(images_normalized, labels, stratify=labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42)

#Data Preprocessing
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=100)
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
X_test_reduced = pca.transform(X_test)

# Calculate explained variance and cumulative variance 
explained_variance = pca.explained_variance_ratio_ # Proportion of total variance explained by each principal componenet
cumulative_variance = np.cumsum(explained_variance) # Cumulative total variance explained by the first 100 components

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA')
plt.grid(True)
plt.show()

# Print the cumulative variance for 100 components
# 100 components explain 90.50% of variance 
print(f'Cumulative variance explained by 100 components: {cumulative_variance[99]:.4f}')

# Clustering Technique
# K-Means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

np.random.seed(42)

k_range = range(1, 21)  # Trying from 1 to 20 clusters

silhouette_scores = []
max_sil_score = -1 
best_k = -1  

# Determine the highest silhouette score and associated number of clusters
for k in k_range[1:]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_train_reduced)
    sil_score = silhouette_score(X_train_reduced, clusters)
    silhouette_scores.append(sil_score)
    
    if sil_score > max_sil_score:
        max_sil_score = sil_score
        best_k = k

# Plot the Silhouette Scores for different k values
plt.figure(figsize=(8, 6))
plt.plot(k_range[1:], silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Number of Clusters')
plt.grid(True)
plt.show()

# The best silhouette score was 0.14516161898358884 for 20 clusters
print(f'Best silhouette score: {max_sil_score}')
print(f'Number of clusters: {k}')

# K-Means clustering with 20 clusters
kmeans = KMeans(n_clusters=20, random_state=42)
kmeans.fit(X_train_reduced)
clusters = kmeans.predict(X_train_reduced)

# Neural Network Architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=20)
y_val_cat = to_categorical(y_val, num_classes=20)
y_test_cat = to_categorical(y_test, num_classes=20)

# Define the model
model = Sequential([
    Dense(128, input_dim=100, activation='relu'),
    Dense(64, activation='relu'),
    Dense(20, activation='softmax')  # Assuming 20 classes
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model and save the history
history = model.fit(X_train_reduced, y_train_cat, validation_data=(X_val_reduced, y_val_cat), epochs=50, batch_size=32)

import matplotlib.pyplot as plt

# Plotting training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Adding the test accuracy point
test_loss, test_accuracy = model.evaluate(X_test_reduced, y_test_cat, verbose=0)
plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy: {test_accuracy:.2f}')
plt.legend(loc='upper left')

plt.show()


#5 Evaluate model
test_loss, test_accuracy = model.evaluate(X_test_reduced, y_test_cat)
print(f"Test Accuracy: {test_accuracy:.2f}")

from sklearn.metrics import classification_report

# Predict on test set
y_pred_cat = model.predict(X_test_reduced)
y_pred = np.argmax(y_pred_cat, axis=1)

print(classification_report(y_test, y_pred))
