import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History
# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Preprocess data
# Transform 2D data to 4D
# Convert pixels to floats
# Normalize to have values between 0 and 1
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32')/ 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') /255
# convert labels to one hot encoded values
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#arg max returns argument with max value in specified axis
original_test_labels = np.argmax(test_labels, axis=1) # Save original labels for confusion matrix

# Define model
model = Sequential([
    #input image
    #output tensor
Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    #input tensor
    #output smaller reduced tensor
MaxPooling2D((2, 2)),
    #input 3D tensor
    #output vector
Flatten(),
    # input vector
    # output 64 vector
Dense(64, activation='relu'),
    # input 64 vector
    # output 10 vector
Dense(10, activation='softmax')
])
# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train model
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1                 )


history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2,
callbacks=[history])
# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
# Confusion matrix
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.show()
# Display 25 images from the test set with their predicted labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(predicted_labels[i])
plt.show()

# most common errors (7,2), (9,7), (9,8)