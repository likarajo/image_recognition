import pickle
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras import models
from random import randint

if not os.path.isdir('models'):
    os.mkdir('models')

def load_data(batch_number):
    with open('cifar-10-batches-py/data_batch_'+ str(batch_number), 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1) # reshape for Conv layer input
    labels = batch['labels']
    return features, labels

batch_1, labels_1 = load_data(1)
batch_2, labels_2 = load_data(2)
batch_3, labels_3 = load_data(3)
batch_4, labels_4 = load_data(4)
batch_5, labels_5 = load_data(5)

# Create train, validation, test sets

X_train = np.append(batch_1, batch_2, axis=0)
X_train = np.append(X_train, batch_3, axis=0)
X_train = np.append(X_train, batch_4, axis=0)
X_train = np.append(X_train, batch_5[0:5000], axis=0)
Y_train = np.append(labels_1, labels_2, axis=0)
Y_train = np.append(Y_train, labels_3, axis=0)
Y_train = np.append(Y_train, labels_4, axis=0)
Y_train = np.append(Y_train, labels_5[0:5000], axis=0)
#print("X_train:", X_train.shape, "Y_train:", Y_train.shape)

X_test = batch_5[5000:]
Y_test = labels_5[5000:]

# Preprocess features

X_train = X_train*1./255
X_test = X_test*1./255

# Create one-hot encoded labels

num_classes = 10
Y_train_one_hot = np_utils.to_categorical(Y_train, num_classes)
Y_test_one_hot = np_utils.to_categorical(Y_test, num_classes)

# Create CNN model

input_conv_neurons = 32
hidden_conv_neurons = [64, 128]
hidden_conv_dropout = [0.25, 0.25]
hidden_dense_neurons = [200, 100]
hidden_dense_dropout = [0.4, 0.3]
output_dense_neurons = num_classes

model = Sequential()

# Input Conv Layer
model.add(Conv2D(input_conv_neurons, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32,32,3))) #row and columns size remains same
model.add(BatchNormalization()) #to ensure there is not much covariance shift from output of preceeding layer

# Hidden Conv Layer(s)
model.add(Conv2D(hidden_conv_neurons[0], kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #reduce row column dimensions to half
model.add(Dropout(hidden_conv_dropout[0])) #inactivate the nodes randomly

model.add(Conv2D(hidden_conv_neurons[1], kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) #reduce row column dimensions to half
model.add(Dropout(hidden_conv_dropout[1])) #inactivate randomly 25% of the nodes

# Flatten out Conv
model.add(Flatten())

# Hidden Dense Layer(s)
model.add(Dense(hidden_dense_neurons[0], activation='relu'))
model.add(Dropout(hidden_dense_dropout[0]))

model.add(Dense(hidden_dense_neurons[1], activation='relu'))
model.add(Dropout(hidden_conv_dropout[1]))

# Output Dense Layer
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Compile model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model

history = model.fit(
    X_train, Y_train_one_hot,
    validation_split=0.33,
    batch_size=50,
    epochs=20,
    shuffle=True,
    verbose=1,
    callbacks=[
        EarlyStopping(
            monitor='val_accuracy', #check for val_accuracy
            patience=3), #stop if there is no improvement for 3 epochs
        ModelCheckpoint(
            'models/model_{val_accuracy:.3f}.h5',
            save_best_only=True, #save the best performing model
            save_weights_only=False,
            monitor='val_accuracy')
    ]
)

# Test model

scores = model.evaluate(
    X_test, Y_test_one_hot,
    batch_size=32,
    verbose=1
)

# Save Model

model.save('models/cifar10.model')

# Print and visualize Accuracy and Loss

print('\nTrain result: train_loss: %.4f - train_accuracy: %.4f ' % (history.history['loss'][-1], history.history['accuracy'][-1]*100))
print('\nValidation result: val_loss: %.4f - val_accuracy: %.4f ' % (history.history['val_loss'][-1], history.history['val_accuracy'][-1]*100))
print('\nTest result: test_loss: %.4f - test_accuracy: %.4f ' % (scores[0], scores[1]*100))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# Make prediction

# Load model
model = models.load_model('models/cifar10.model')

# Give name for labels
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def make_prediction(number):
    test_image = np.expand_dims(X_test[number], axis=0)
    test_result = model.predict_classes(test_image)
    return test_result[0]

plt.figure(figsize=(5,9))
for p in range(0, 6):
    x = randint(0, 5000)
    dict_key = make_prediction(x)
    print("{}: Predicted: {}, Actual: {}".format(x, labels[dict_key], labels[Y_test[x]]))
    plt.subplot(3,2,p+1)
    plt.imshow(X_test[x])
    plt.title("Predicted: {}, Actual: {}".format(labels[dict_key], labels[Y_test[x]]))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.show()
