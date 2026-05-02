# import tensorflow as tf
# from keras.datasets import mnist
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# # Load dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize + reshape
# x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# # One-hot encoding
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)

# # -------------------------------
# # 🔥 MODERN DATA AUGMENTATION
# # -------------------------------
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomRotation(0.1),
#     tf.keras.layers.RandomTranslation(0.1, 0.1)
# ])

# # -------------------------------
# # CNN MODEL
# # -------------------------------
# model = Sequential([
#     Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),

#     Conv2D(64, (3,3), activation="relu"),
#     BatchNormalization(),
#     MaxPooling2D((2,2)),

#     Conv2D(128, (3,3), activation="relu"),
#     BatchNormalization(),

#     Flatten(),
#     Dense(128, activation="relu"),
#     Dropout(0.4),
#     Dense(10, activation="softmax")
# ])

# # Compile
# model.compile(
#     optimizer="adam",
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # -------------------------------
# # TRAINING (FIXED WAY)
# # -------------------------------
# model.fit(
#     data_augmentation(x_train),
#     y_train,
#     epochs=15,
#     batch_size=64,
#     validation_data=(x_test, y_test)
# )

# # Evaluate
# loss, acc = model.evaluate(x_test, y_test)
# print("Test Accuracy:", acc)

# # Save model
# model.save("mnist_cnn.h5")
# print("Model saved successfully 🚀")


import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize + reshape
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------------
# ✅ SPLIT TRAIN → TRAIN + VALIDATION
# -------------------------------
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)

# -------------------------------
# 🔥 DATA AUGMENTATION (CORRECT WAY)
# -------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

# -------------------------------
# CNN MODEL (AUGMENTATION INSIDE)
# -------------------------------
model = Sequential([
    data_augmentation,

    Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# TRAINING (CORRECT)
# -------------------------------
model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_data=(x_val, y_val)
)

# -------------------------------
# FINAL EVALUATION (ONLY ON TEST)
# -------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Save model
model.save("mnist_cnn.h5")
print("Model saved successfully 🚀")
