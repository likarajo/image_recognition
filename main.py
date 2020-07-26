import pickle
from random import randint
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import models
import os

# Import data

def load_data(batch_number):
    with open('cifar-10-batches-py/data_batch_'+ str(batch_number), 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data']
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
print("X_train:", X_train.shape, "Y_train:", Y_train.shape)

X_test = batch_5[5000:]
Y_test = labels_5[5000:]

# Preprocess features

X_train = X_train*1./255
X_test = X_test*1./255

# Create one-hot encoded labels

num_classes = 10
Y_train_one_hot = np_utils.to_categorical(Y_train, num_classes)
Y_test_one_hot = np_utils.to_categorical(Y_test, num_classes)

# Create model

classifier = Sequential()
classifier.add(Dense(32, activation='relu', input_dim=3072))
classifier.add(Dense(16, activation='relu'))
classifier.add(Dense(num_classes, activation='softmax'))
classifier.summary()

# Compile model

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model

history = classifier.fit(X_train, Y_train_one_hot, validation_split=0.33,
                         batch_size=200, epochs=2,
                         shuffle=True, verbose=1)

# Test model

scores = classifier.evaluate(X_test, Y_test_one_hot, batch_size=100)

# Accuracy
print('\nTrain result: train_loss: %.4f - train_accuracy: %.4f ' % (history.history['loss'][-1], history.history['accuracy'][-1]*100))
print('\nValidation result: val_loss: %.4f - val_accuracy: %.4f ' % (history.history['val_loss'][-1], history.history['val_accuracy'][-1]*100))
print('\nTest result: test_loss: %.4f - test_accuracy: %.4f ' % (scores[0], scores[1]*100))

# Save Model

if not os.path.exists('models'):
    os.makedirs('models')
classifier.save('models/cifar10.model')

# summarize history for accuracy and loss
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

'''
# Merge inputs and targets
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((Y_train_one_hot, Y_test_one_hot), axis=0)
# Use K-Fold Cross validation\
num_folds = 5
acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits=num_folds, shuffle=True)
# Create models using K-Fold
fold_no = 1
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    model = Sequential()
    model.add(Dense(16, activation='relu', input_dim=3072))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # Fit data to model
    history = model.fit(inputs[train], targets[train], batch_size=50, epochs=50, verbose=1)
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Increase fold number
    fold_no = fold_no + 1
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
'''


# Make prediction

# Load model
model = models.load_model('models/cifar10.model')

# Transpose data for viewing image
X_test1 = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0,2,3,1)

# Give name for labels
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def make_prediction(number):
    test_image = np.expand_dims(X_test[number], axis=0)
    test_result = model.predict_classes(test_image)
    dict_key = test_result[0]
    print("{}: Predicted: {}, Actual: {}".format(number, labels[dict_key], labels[Y_test[number]]))

    plt.figure(figsize=(8, 8))
    plt.imshow(X_test1[number])
    plt.title("Predicted: {}, Actual: {}".format(labels[dict_key], labels[Y_test[number]]))
    plt.show()

for _ in range(0, 6):
    x = randint(0, 5000)
    make_prediction(x)
