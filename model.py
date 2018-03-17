import os
import csv
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt

def readCSVFile(filename):
	# Read the lines of a csv file produced by the Udacity mac_sim.app
	# Remove the first line as this contains the title of the individual items of the following lines,
	# i.e. no images or measurements

    lines = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    # Remove the first line
    lines = lines[1:]

    return(lines)

def makeListOfPathsAndMeasurements(my_input_lines, correction=0.2):
	# Take a list of lines from a csv file and produce a list of (image path, angle measurement) tuples
	# The correction factor function parameter is applied to the center measurement to produce 
	# adjusted measurements for the left and the right images.

    measurements = []
    imagePaths = []
    
    # Extract center, left and right image paths as well as the associated measurement from each line
    for line in my_input_lines:
        center_image = line[0]
        left_image = line[1].strip()
        right_image = line[2].strip()
        steering_value = float(line[3])
    
    	# Apply the correction parameter to create artificial measurements for the left and right images
        measurements.append(steering_value)
        measurements.append(steering_value + correction)
        measurements.append(steering_value - correction)
        
        imagePaths.append(center_image)
        imagePaths.append(left_image)
        imagePaths.append(right_image)
               
    return(list(zip(imagePaths, measurements)))

def nVidiaModel():
	# A Keras implementation of the nVidia CNN architecture, including preprocessing layers

    model = Sequential()
    # Preprocessing: normalize and crop images
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    
    # The CNN layers
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    # The dense layers
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1)) 
    
    return model

def generator(samples, batch_size=32):
	# A generator function for usage by Keras model.fit_generator
	# Note this function will generate batches which are twice the size of the batch_size parameter
    # because it also flips each image.
	
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = cv2.imread(name)

				# Convert from BGR to RGB format. cv2.imread returns BGR, but the simulator produces RGB.      
                b,g,r = cv2.split(image)
                rgb_image = cv2.merge([r,g,b])

                angle = float(batch_sample[1])
                images.append(rgb_image)
                angles.append(angle)
                
                # Flip image and produce the second image
                images.append(cv2.flip(rgb_image,1))
                angles.append(angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Here follows the action
my_csv_lines1 = readCSVFile('./mydata/driving_log1.csv')
my_csv_lines2 = readCSVFile('./mydata/driving_log2.csv')

my_csv_lines = my_csv_lines1 + my_csv_lines2
print("Number of lines read from my driving_log.csv files: " + str(len(my_csv_lines)))

samples = makeListOfPathsAndMeasurements(my_csv_lines, 0.25)

# Split the samples into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("Number of training samples: " + str(len(train_samples)))
print("Number of validation samples: " + str(len(validation_samples)))

# Generator functions for training and validation
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = nVidiaModel()

# Compile and train the model
model.compile(loss='mse', optimizer='adam')

# Note the 2* on the number of training and validation samples. This is because the generator flips each image
# and therefore generates batches of twice the size of the function parameter.
history = model.fit_generator(train_generator, samples_per_epoch=(2*len(train_samples)),
                              validation_data=validation_generator, nb_val_samples=(2*len(validation_samples)),
                              nb_epoch=3, verbose=1)

# And save the result
model.save('model.h5')

# Lets look at the results of the fit_generator
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()