"""
Created on Tue Sep  8 17:15:23 2020

CONVOLUTIONAL NEURAL NETWORK FOR DIGIT RECOGNITION

@author: Marc
"""

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

# Load and prepare the train and test set
def load_dataset():
    # Load the dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # Reshape the dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # One hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# Scale pixels
def prep_pixels(train, test):
    # Convert from integers to float
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # Normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

# Define the CNN classifier
def define_classifier():
    # Build the structure
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D((2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(100, activation = 'relu'))
    classifier.add(Dense(10, activation = 'softmax'))
    # Compile the model
    classifier.compile(optimizer = SGD(lr = 0.01, momentum = 0.9), loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    return classifier

# Evaluate the classifier using the K-Fold Cross-Validation
def evaluate_classifier(dataX, dataY, n_folds = 5):
    scores, histories = list(), list()
    # Prepare Cross-Validation
    kfold = KFold(n_folds, shuffle = True, random_state = 1)
    # Enumerate splits
    for trainX_i, testX_i in kfold.split(dataX):
        # Define classifier
        classifier = define_classifier()
        # Select rows for train and test
        trainX, trainY, testX, testY = dataX[trainX_i], dataY[trainX_i], dataX[testX_i], dataY[testX_i]
        # Fit the classifier
        history = classifier.fit(trainX, trainY, batch_size = 32, epochs = 10, 
                                 validation_data = (testX, testY), verbose = 1)
        # Evaluate the classifier
        _, acc = classifier.evaluate(testX, testY, verbose = 1)
        print('> ACC: %.3f' % (acc * 100.0))
        # Store history, accuracy
        scores.append(acc)
        histories.append(history)
    return scores, histories

# Plot learning curves
def visualise_learning(histories):
    for i in range(len(histories)):
        # Plot LOSS
        plt.subplot(2, 1, 1)
        plt.title('Cross-Entropy Loss')
        plt.plot(histories[i].history['loss'], color = 'blue', label = 'train')
        plt.plot(histories[i].history['val_loss'], color = 'orange', label = 'test')
        # Plot ACCURACY
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color = 'blue', label = 'train')
        plt.plot(histories[i].history['val_accuracy'], color = 'orange', label = 'test')
    plt.show()

# Summarize classifier performance
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    
# Run all parts together
def run():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    scores, histories = evaluate_classifier(trainX, trainY)
    visualise_learning(histories)
    summarize_performance(scores)
  
def save_model():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    classifier = define_classifier()
    classifier.fit(trainX, trainY, epochs = 10, batch_size = 32, verbose = 1)
    classifier.save('final_classifier.h5')
    
##############################################################################################################

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image('image.png')
	# load model
	model = load_model('final_classifier.h5')
	# predict the class
	digit = model.predict_classes(img)
	print(digit[0])

# entry point, run the example
run_example()