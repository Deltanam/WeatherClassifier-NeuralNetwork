import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential, datasets, layers, models
import cv2
import os

# image preprocessing and setting targets
# A: Formatting 
path = "D:\Python\Weather_Project\dataset2"
rimg_list = []
targets = []
class_names = ['cloudy', 'rain', 'shine', 'sunrise']
inputwidth = 64

for filename in os.listdir(path):
    # B: target code
    for idx, val in enumerate(class_names):
        if(filename.find(val) != -1):
            targets.append(idx)
            break
    
    # C: read file -> correct colour -> resize
    img = cv2.imread(os.path.join(path,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # correct colour -> no greyscale
    rimg = cv2.resize(img, (inputwidth,inputwidth)) / 255.0 # normalize as well as resize
    rimg_list.append(rimg)

rimg_list = np.array(rimg_list)
targets = np.array(targets)

# Set a seed for repeatability
np.random.seed(seed=0)

# D: Split data (80%, 20%)
permuted_index = np.random.permutation(rimg_list.shape[0])
index_80 = (int)(rimg_list.shape[0] * 0.80)
X_train = rimg_list[permuted_index[0:index_80]]
Y_train = targets[permuted_index[0:index_80]]
X_test = rimg_list[permuted_index[index_80+1:]]
Y_test = targets[permuted_index[index_80+1:]]

# E: Create machine learning architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(inputwidth, inputwidth, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))

model.compile(optimizer='adam', 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_train, Y_train))


model.summary()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Apply Test Data
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)

print('Accuracy = ',test_acc)

# Confusion Matrix ---------------------------
# Pre-defined compute_confusion_matrix function provided to us
def compute_confusion_matrix(true, pred):
  K = len(np.unique(true)) # Number of classes
  result = np.zeros((K, K))
  for i in range(len(true)):
    result[true[i]][pred[i]] += 1
  return result

# Confusion Matrix
Y_predict = model.predict(X_test)
prediction_probability_vector = []
for i in range(Y_predict.shape[0]):
  prediction_probability_vector.append(np.argmax(Y_predict[i]))

confusion_matrix = compute_confusion_matrix(Y_test, prediction_probability_vector)
print('The Confusion Matrix is: \n', confusion_matrix)


# Personal Test Data ------------------------------
path2 = "D:\Python\Weather_Project\TestPhotos"
rimg_list_personal = []
targets_personal = []

for filename in os.listdir(path2):
    # B: target code
    for idx, val in enumerate(class_names):
        if(filename.find(val) != -1):
            targets_personal.append(idx)
            break
    
    # C: read file -> correct colour -> resize
    img = cv2.imread(os.path.join(path2,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # correct colour -> no greyscale
    rimg = cv2.resize(img, (inputwidth,inputwidth)) / 255.0 # normalize as well as resize
    rimg_list_personal.append(rimg)

    prediction = model.predict(np.array(rimg_list_personal))
    print(filename, ' is predicted as: ', class_names[np.argmax(prediction[-1])])

rimg_list_personal = rimg_list_personal
rimg_list_personal = np.array(rimg_list_personal)
targets_personal = np.array(targets_personal)

test_loss, test_acc = model.evaluate(rimg_list_personal,  targets_personal, verbose=2)
print('Personal photos accuracy = ', test_acc)