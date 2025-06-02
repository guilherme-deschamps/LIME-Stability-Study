# This is a sample Python script.

# Press F6 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import gray2rgb, rgb2gray
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image
import numpy as np
import os

train_path = 'dataset/splitted_dataset/train'
val_path = 'dataset/splitted_dataset/val - 20% data'
IMG_HEIGHT = 200
IMG_WIDTH = 300
BATCH_SIZE = 32
CHECKPOINT_PATH = "best model - 88.6 accuracy/cp.ckpt"


def read_dataset():
    """Reads the dataset, and returns a training set and a validation set."""
    print('Reading dataset')
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     batch_size=BATCH_SIZE,
                                                     class_mode='sparse')
    val_set = val_datagen.flow_from_directory(val_path,
                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                              batch_size=BATCH_SIZE,
                                              class_mode='sparse')

    x_test = []

    for folder in os.listdir(val_path):
        sub_path = val_path + "/" + folder

        print('Reading images from sub-path {}'.format(sub_path))
        for img in os.listdir(sub_path):
            image_path = sub_path + "/" + img
            img_arr = cv2.imread(image_path)
            x_test.append(img_arr)
        print('Finished reading {}'.format(folder))

    print('Creating numpy arrays for images')

    return training_set, val_set


def create_model():
    """
    Creates the model and defines its architecture.
    :return: The compiled (non-trained) model.
    """

    # Creates the basic structure for the model
    model = models.Sequential()

    # Adds the Convolutional and the Max Pooling layers to the model.
    # The input_shape parameter defines the shape of the images that will come into the first layer.
    model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, 3, activation='relu'))

    # Flattens the input that comes from the previous layer, and adds a Dense later afterwards.
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Output layer with 11 output nodes, since the dataset contains 11 classes.
    model.add(layers.Dense(11))

    # Compile the model.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def plot_model_performance(history):
    """Generates plots for the accuracy and the loss of the model over the epochs.
    Receives as parameter the history os the trained model."""
    print('Starting plots')

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print('Plots finished')


# Loads the datasets
training_set, val_set = read_dataset()

# Creates the model and loads the trained weights
model = create_model()
loss, acc = model.evaluate(val_set, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(CHECKPOINT_PATH)
loss, acc = model.evaluate(val_set, verbose=2)
print("Weights loaded, accuracy: {:5.2f}%".format(100 * acc))

# # Calculating metrics from prediction
# Y_test_preds = model.predict(val_set)
#
# print("Test Accuracy : {}".format(accuracy_score(val_set.classes, Y_test_preds)))
# print("\nConfusion Matrix : ")
# print(confusion_matrix(val_set.classes, Y_test_preds))
# print("\nClassification Report :")
# print(classification_report(val_set.classes, Y_test_preds, target_names=val_set.class_indices))

# Explaining prediction with LIME
explainer = lime_image.LimeImageExplainer(random_state=123)

x_train, y_train = next(training_set)
x_test, y_test = next(val_set)
mapping = {v: k for k, v in val_set.class_indices.items()}

rng = np.random.RandomState(42)


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def jaccard_distance(usecase):
    sim = []
    for l in usecase:
        i_sim = []
        for j in usecase:
            i_sim.append(1 - jaccard_similarity(l, j))
        sim.append(i_sim)
    return sim


def plot_explanation(explanation, filepath, features, idx, pred):
    """
    Method responsible for plotting the information of an explanation
    :param explanation: explanation to be plotted
    :param filepath: path to save the image of the generated plot
    :param features: amount of features to add in the explanation image
    :return:
    """
    # Obtaining image and mask that displays most important parts of image
    img, mask = explanation.get_image_and_mask(y_test[idx], positive_only=False, hide_rest=False, num_features=features,
                                               min_weight=0.03)

    # Select the same class explained on the figures above
    ind = explanation.top_labels[0]

    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    # Plot figure with the images of the explanations
    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 8))
    fig.suptitle('LIME explanation\n\n'
                 'Actual specie: {}; Predicted specie: {}'
                 .format(mapping[y_test[idx]], mapping[pred]), fontsize=20)
    ax.imshow(x_test[idx], cmap="gray")
    ax.set_title("Original image")
    ax.axis('off')
    ax1.imshow(mark_boundaries(img, mask))
    ax1.set_title("Both positive and negative contributions")
    ax1.axis('off')
    ax2.set_title("Heatmap displaying weight of each piece of the image")
    ax2.axis('off')
    ax2.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    # ax2.colorbar()

    # Saving plot to file on correct folder
    plt.savefig(filepath)


# Neighbourhood size. Number of random perturbations to be used when generating the explanations.
# References: 100 = Lee et al.; 1000 = Graziani et al.; 5000 = Zafar & Khan
num_samples = [100, 1000, 5000]
# Maximum amount of superpixels to be in the explanation images. References: 10 and 100 = Graziani et al.
num_features = [10]


# Number of experiments to run
def generate_explanations():
    """
    Function responsible for generating 10 explanations using each configuration of LIME.
    """
    for experiment in range(5):
        # idx = rng.choice(range(len(x_test)))
        idx = 22
        pred = make_prediction(idx)
        for s in num_samples:
            for f in num_features:
                path = "results/run {}/{} samples/".format(experiment + 1, s)
                weights = []
                # Runs 10 times each configuration
                for x in range(10):
                    explanation = explainer.explain_instance(x_test[idx].astype('double'), model.predict, random_seed=123,
                                                             top_labels=5, num_samples=s)
                    filepath = path + 'explanation {}.png'.format(x)
                    plot_explanation(explanation, filepath, f, idx, pred)

                    # Round weights to 2 digits just like in Lee et al.
                    original_weights = explanation.local_exp[explanation.top_labels[0]]
                    rounded_weights = []
                    for i in range(len(original_weights)):
                        w = round(original_weights[i][1], 2)
                        index = original_weights[i][0]
                        rounded_weights.append((index, w))
                    weights.append(rounded_weights)

                sim = jaccard_distance(weights)
                np.savetxt(path + 'jaccard_distances.csv', sim, delimiter=",")
                print(np.asarray(sim).mean())

                plt.matshow(sim)
                plt.colorbar()
                plt.savefig(path + 'jaccard_distances.png', bbox_inches='tight')
                plt.show()


def make_prediction(idx):
    """
    Makes a prediction for the informed sample.
    :return: the prediction made
    """

    print("Actual Target Value     : {}".format(mapping[y_test[idx]]))
    pred = model.predict(x_test[idx:idx + 1]).argmax(axis=1)[0]
    print("Predicted Target Values : {}".format(mapping[pred]))
    return pred


# generate_explanations()
