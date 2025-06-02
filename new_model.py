# import cv2
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import os
#
# train_path = 'dataset/splitted_dataset/train'
# val_path = 'dataset/splitted_dataset/val - 20% data'
# IMG_HEIGHT = 200
# IMG_WIDTH = 300
# BATCH_SIZE = 32
#
#
# def read_dataset():
#     """Reads the dataset, and returns a training set and a validation set."""
#     print('Reading dataset')
#     train_datagen = ImageDataGenerator(rescale=1. / 255)
#     val_datagen = ImageDataGenerator(rescale=1. / 255)
#
#     training_set = train_datagen.flow_from_directory(train_path,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                      batch_size=BATCH_SIZE,
#                                                      class_mode='sparse')
#     val_set = val_datagen.flow_from_directory(val_path,
#                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                               batch_size=BATCH_SIZE,
#                                               class_mode='sparse')
#
#     x_test = []
#
#     for folder in os.listdir(val_path):
#         sub_path = val_path + "/" + folder
#
#         print('Reading images from sub-path {}'.format(sub_path))
#         for img in os.listdir(sub_path):
#             image_path = sub_path + "/" + img
#             img_arr = cv2.imread(image_path)
#             x_test.append(img_arr)
#         print('Finished reading {}'.format(folder))
#
#     print('Creating numpy arrays for images')
#
#     return training_set, val_set
#
#
# def create_model():
#     """Creates the model and defines its architecture to be loaded, then returns it."""
#     print('Creating model')
#     model = models.Sequential()
#
#     model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, 3, activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(128, 3, activation='relu'))
#
#     model.add(layers.Flatten())
#     model.add(layers.Dense(128, activation='relu'))
#
#     # Output layer should have 11 output nodes, since the dataset has 11 classes
#     model.add(layers.Dense(11))
#
#     # Compile the model
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
#     print('Model created!')
#     return model
#
#
# def plot_model_performance(history):
#     """Generates plots for the accuracy and the loss of the model over the epochs.
#     Receives as parameter the history os the trained model."""
#     print('Starting plots')
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
#     fig.suptitle('Accuracy and loss of classification model', fontsize=20)
#
#     # summarize history for accuracy
#     ax1.plot(history.history['accuracy'])
#     ax1.plot(history.history['val_accuracy'])
#     ax1.title('model accuracy')
#     ax1.ylabel('accuracy')
#     ax1.xlabel('epoch')
#     ax1.legend(['train', 'test'], loc='upper left')
#
#     # summarize history for loss
#     ax2.plot(history.history['loss'])
#     ax2.plot(history.history['val_loss'])
#     ax2.title('model loss')
#     ax2.ylabel('loss')
#     ax2.xlabel('epoch')
#     ax2.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     print('Plots finished')
#
#
# # Loads the datasets
# training_set, val_set = read_dataset()
#
# # Creates the model and loads the trained weights
# model = create_model()
# loss, acc = model.evaluate(val_set, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
#
# # Train model
# # steps_per_epoch = how many steps (iterations) define an epoch. Equals to size of dataset / batch size
# print('Starting training model')
# history = model.fit(
#     training_set,
#     steps_per_epoch=1818 // BATCH_SIZE,
#     validation_data=val_set,
#     validation_steps=459 // BATCH_SIZE,
#     epochs=10)
# print('Model finished training')
