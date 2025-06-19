import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

categories = ["cat", "dog"]  # Update categories to match directories

# Load preprocessed test data
X_test = np.load("../X_test.npy")
y_test = np.load("../y_test.npy")

# Load model
model = joblib.load("F:/vscode/python/task3/svm_cats_vs_dogs.pkl")

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Visualize predictions
for i in range(5):
    plt.imshow(X_test[i].reshape(64, 64, 3).astype("uint8"))
    plt.title(f"Predicted: {categories[y_pred[i]]}, Actual: {categories[y_test[i]]}")
    plt.show()
