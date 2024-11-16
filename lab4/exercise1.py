import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
# Standardize data to have a mean of 0 and a standard deviation of 1
# f(x) = (x - mean) / standard deviation
X_scaled = scaler.fit_transform(X)
# OneHotEncoder transforms a list of categorical classes into a binary matrix with values 0 and 1.
# Each unique class is represented as a separate column.
# Each row corresponds to an object and contains a 1 in the column of its class, with 0s elsewhere.
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# The entry layer has 4 neurons, as the length of X.shape[1] is 4.
# This corresponds to the 4 features in the dataset.
# The last layer has 3 neurons because y_encoded.shape[1] is 3.
# This indicates that the irises are divided into 3 classes.
model = Sequential([
    # I tried ReLU, Swish, Tanh, and Hard Sigmoid activation functions for the hidden layers, and all of them
    # achieved an accuracy of over 90%. I consider this result to be very high.
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax'),
])

# Different optimizers can result in different training times and performance.
# For example, changing the optimizer to SGD may reduce the accuracy to 80%.
# Yes, we can modify the learning rate for optimizers to control how quickly the model learns.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# The default batch size in Keras is 32. This means that the model processes 32 samples at a time during training.
# When I changed the batch size to smaller values like 4 or 8, the loss curve became less stable.
# This is because smaller batch sizes lead to more frequent weight updates, causing the training process to be noisier.
# While the model may converge faster, the loss curve tends to fluctuate more, as the updates are based on fewer samples.
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc *100)

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='lightgrey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='lightgrey')
plt.legend()
plt.tight_layout()
plt.show()

model.save('iris_model.h5')

plot_model(model, to_file='iris_model.png', show_shapes=True, show_layer_names=True)


