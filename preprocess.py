import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_images(dataset_path, img_size=64):
    # Update categories
    categories = ["cat", "dog"]
    data = []
    labels = []

    for category in categories:
        path = os.path.join(dataset_path, category)
        label = categories.index(category)  # 0 for cat, 1 for dog
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_img = cv2.resize(img_array, (img_size, img_size))
                data.append(resized_img.flatten())  # Flatten image
                labels.append(label)
            except Exception as e:
                print(f"Error processing image {img}: {e}")

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    dataset_path = "F:/vscode/python/task3/dataset"  # Path to the dataset folder
    data, labels = load_and_preprocess_images(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    np.save("../X_train.npy", X_train)
    np.save("../X_test.npy", X_test)
    np.save("../y_train.npy", y_train)
    np.save("../y_test.npy", y_test)
    print("Preprocessing complete. Data saved.")
