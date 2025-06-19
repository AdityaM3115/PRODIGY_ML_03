Cats vs Dogs Classification Using SVM
This project classifies images of cats and dogs using a Support Vector Machine (SVM). Below are the details of the files and processes involved.

File Descriptions
preprocess.py:

Preprocesses raw images into flattened grayscale vectors for model training.

Resizes images, converts them to grayscale, and saves the processed data as:

X_train.npy: Numpy array of feature vectors.

y_train.npy: Numpy array of corresponding labels (0 = cat, 1 = dog).

train.py:

Trains an SVM model using the preprocessed data (X_train.npy and y_train.npy).

Saves the trained model as svm_cats_vs_dogs.pkl.

test.py:

Loads the trained SVM model from svm_cats_vs_dogs.pkl.

Evaluates the model on test/validation data.

Displays metrics such as accuracy and confusion matrix.

X_train.npy:

Numpy file containing preprocessed feature vectors for training.

y_train.npy:

Numpy file containing labels (0 for cats, 1 for dogs) corresponding to X_train.npy.

svm_cats_vs_dogs.pkl:

Pickle file storing the trained SVM model.

Can be loaded for making predictions on new images.

Preprocessing Details
Resizing: All images are resized to a uniform size (e.g., 64x64 pixels).

Grayscale Conversion: Images are converted to grayscale to reduce feature dimensions.

Flattening: Each image is flattened into a 1D array for model input.

Normalization: Pixel values are scaled to the range [0, 1] for uniformity.
