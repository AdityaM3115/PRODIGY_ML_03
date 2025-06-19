import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
import joblib

# Load preprocessed data
print("Loading data...")
X_train = np.load("../X_train.npy")
y_train = np.load("../y_train.npy")
print(f"Data loaded: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Shuffle the data (optional, for better randomness in batches)
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# Create batches for training
batch_size = 1000  # Adjust based on available memory and data size
num_batches = (len(X_train) + batch_size - 1) // batch_size

# Initialize model
model = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))

# Train in batches
print("Training SVM in batches...")
for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(X_train))
    print(f"Training batch {batch_idx + 1}/{num_batches} (samples {start} to {end})...")
    
    # Fit on the current batch
    model.fit(X_train[start:end], y_train[start:end])

print("Training complete.")

# Save model
print("Saving model...")
joblib.dump(model, "../svm_cats_vs_dogs.pkl")
print("Model saved as svm_cats_vs_dogs.pkl")
