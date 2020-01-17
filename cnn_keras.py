import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D

classifier = Sequential()

classifier.add(Conv2D(32, (3,3),input_shape = (64,64,3), activation='relu'))

classifier.add(MaxPooling2D(pool_size = (2,2), strides = 2))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#compile the whole cnn through the use of the optimizer
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip=True)

test_datgen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')

test_set = test_datgen.flow_from_directory('dataset/test_set',
                                           target_size = (64,64),
                                           batch_size = 32,
                                           class_mode = 'binary')
classifier.fit_generator(train_set, samples_per_epoch = 8000, nb_epoch = 25, validation_data= test_set, nb_val_samples = 2000)
