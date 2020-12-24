"""
Authors: Madelyn, Nyjay, Wilbert
"""

import numpy as np
import tensorflow as tf
from os import listdir
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, losses, metrics, Model
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report

LABELS = listdir("asl_data/train")
NUM_LABELS = len(LABELS)
INPUT_SHAPE = (200, 200, 3)
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.1
NUM_NODES = 256
METRICS = [metrics.BinaryAccuracy(), 
           metrics.Precision(),
           metrics.Recall()]


def get_baseline_model(print_summary=True):
    """
    Builds and compiles TensorFlow model with one Dense sigmoid node.
    """
    baseline_model = tf.keras.Sequential()
    baseline_model.add(layers.Flatten(input_shape=INPUT_SHAPE))
    baseline_model.add(layers.Dense(NUM_LABELS, activation="softmax"))
    baseline_model.compile(loss=losses.CategoricalCrossentropy(), 
                           optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), 
                           metrics = METRICS)
    if print_summary:
        baseline_model.summary()
    return baseline_model

def get_inception_model(trainable=False, num_nodes=NUM_NODES, 
                        dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE, 
                        print_summary=True):
    """
    Builds and compiles InceptionV3 transfer learning model.
    """
    transfer_model = InceptionV3(input_shape=INPUT_SHAPE, include_top = False, # leave out the last fully connected layer
                           weights = 'imagenet')
    for layer in transfer_model.layers:
        layer.trainable = trainable
    hidden_layers = []
    hidden_layers.append(layers.Flatten()(transfer_model.output))
    hidden_layers.append(layers.Dense(num_nodes, activation = "relu")(hidden_layers[-1]))
    hidden_layers.append(layers.Dropout(dropout_rate)(hidden_layers[-1]))
    output_layer = layers.Dense(NUM_LABELS, activation="softmax")(hidden_layers[-1])
    model1 = Model(transfer_model.input, output_layer)
    model1.compile(loss=losses.CategoricalCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)
    if print_summary:
        model1.summary()
    return model1

def get_resnet50_model(trainable=False, num_nodes=NUM_NODES, 
                       dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE, 
                       print_summary=True):
    """
    Builds and compiles ResNet50 transfer learning model.
    """
    transfer_model = ResNet50(input_shape=INPUT_SHAPE, include_top = False, # leave out the last fully connected layer
                           weights = 'imagenet')
    for layer in transfer_model.layers:
        layer.trainable = trainable
    hidden_layers = []
    hidden_layers.append(layers.Flatten()(transfer_model.output))
    hidden_layers.append(layers.Dense(num_nodes, activation = "relu")(hidden_layers[-1]))
    hidden_layers.append(layers.Dropout(dropout_rate)(hidden_layers[-1]))
    output_layer = layers.Dense(NUM_LABELS, activation="softmax")(hidden_layers[-1])
    model2 = Model(transfer_model.input, output_layer)
    model2.compile(loss=losses.CategoricalCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)
    if print_summary:
        model2.summary()
    return model2

def get_vgg16_model(trainable=False, num_nodes=NUM_NODES, 
                    dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE, 
                    print_summary=True):
    """
    Builds and compiles VGG16 transfer learning model.
    """
    transfer_model = VGG16(input_shape=INPUT_SHAPE, include_top = False, # leave out the last fully connected layer
                           weights = 'imagenet')
    for layer in transfer_model.layers:
        layer.trainable = trainable
    hidden_layers = []
    hidden_layers.append(layers.Flatten()(transfer_model.output))
    hidden_layers.append(layers.Dense(num_nodes, activation = "relu")(hidden_layers[-1]))
    hidden_layers.append(layers.Dropout(dropout_rate)(hidden_layers[-1]))
    output_layer = layers.Dense(NUM_LABELS, activation="softmax")(hidden_layers[-1])
    model3 = Model(transfer_model.input, output_layer)
    model3.compile(loss=losses.CategoricalCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)
    if print_summary:
        model3.summary()
    return model3

def print_results(y_test, predictions, class_report=False):
    y_pred = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    conf_mat = confusion_matrix(y_test, y_pred)
    if class_report:
        print(classification_report(y_test, y_pred, target_names=LABELS))
    else:
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1)
        #print("Confusion Matrix: ", conf_mat)
        #print(str(accuracy) + " | " + str(precision) + " | " + str(recall) + " | " + str(f1) + " |")
    return accuracy, precision, recall, f1, conf_mat
    