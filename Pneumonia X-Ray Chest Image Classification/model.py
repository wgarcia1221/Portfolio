"""
Authors: Madelyn, Nyjay, Wilbert
"""

import numpy as np
import tensorflow as tf
from os import listdir
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, losses, metrics, Model
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score, classification_report, roc_curve

LABELS = listdir("chest_xray/train")
NUM_LABELS = len(LABELS)
INPUT_SHAPE = (256, 256, 3)
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.25
NUM_NODES = 512
METRICS = [metrics.BinaryAccuracy(), 
           metrics.Precision(),
           metrics.Recall()]


def get_baseline_model(learning_rate=LEARNING_RATE, print_summary=False):
    """
    Builds and compiles TensorFlow model with one Dense sigmoid node.
    """
    baseline_model = tf.keras.Sequential()
    baseline_model.add(layers.Flatten(input_shape=INPUT_SHAPE))
    baseline_model.add(layers.Dense(1, activation="sigmoid"))
    baseline_model.compile(loss=losses.BinaryCrossentropy(), 
                           optimizer=optimizers.Adam(learning_rate=learning_rate), 
                           metrics = METRICS)
    if print_summary:
        baseline_model.summary()
    return baseline_model

def get_proj3_model(learning_rate=LEARNING_RATE, print_summary=False):
    """
    Builds and compiles our TensorFlow model from Project 3.
    """
    model1 = tf.keras.Sequential()
    model1.add(layers.Conv2D(filters = 10, kernel_size = (3,3), strides = (1,1), 
                             activation = 'relu', input_shape=INPUT_SHAPE))
    model1.add(layers.Conv2D(filters = 10, kernel_size = (3,3), strides = (1,1), 
                             activation = 'relu'))
    model1.add(layers.Flatten())
    model1.add(layers.Dense(64, activation="relu"))
    model1.add(layers.Dense(1, activation="sigmoid"))
    model1.compile(loss=losses.BinaryCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)
    if print_summary:
        model1.summary()
    return model1

def get_kaggle_model(learning_rate=LEARNING_RATE, print_summary=False):
    """
    Builds and compiles a model inspired by FIXME notebook on Kaggle;
    uses batch normalization and max pool layers in addition to Conv2D 
    layers.
    """
    model2 = tf.keras.Sequential()

    model2.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides = (1,1), 
                             activation = 'relu', input_shape=(256,256,3)))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPool2D((2,2), strides = 2))

    model2.add(layers.Conv2D(filters = 128, kernel_size = (3,3), 
                             strides = 1, activation = 'relu'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPool2D((2,2), strides = 2))

    model2.add(layers.Conv2D(filters = 64, kernel_size = (3,3), 
                             strides = 1, activation = 'relu'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPool2D((2,2), strides = 2))

    model2.add(layers.Conv2D(filters = 32, kernel_size = (3,3), 
                             strides = 1, activation = 'relu'))
    model2.add(layers.BatchNormalization())
    model2.add(layers.MaxPool2D((2,2), strides = (2,2)))

    model2.add(layers.Flatten())     
    model2.add(layers.Dense(128, activation="relu"))
    model2.add(layers.Dropout(.5))

    model2.add(layers.Dense(1, activation="sigmoid"))
    model2.compile(loss=losses.BinaryCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)

    if print_summary:
        model2.summary()
    return model2

def get_inception_model(trainable=False, num_nodes=NUM_NODES, 
                       dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE, 
                       print_summary=False):
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
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layers[-1])
    model5 = Model(transfer_model.input, output_layer)
    model5.compile(loss=losses.BinaryCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)
    if print_summary:
        model5.summary()
    return model5

def get_resnet50_model(trainable=False, num_nodes=NUM_NODES, 
                       dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE, 
                       print_summary=False):
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
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layers[-1])
    model5 = Model(transfer_model.input, output_layer)
    model5.compile(loss=losses.BinaryCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)
    if print_summary:
        model5.summary()
    return model5

def get_vgg16_model(trainable=False, num_nodes=NUM_NODES, 
                    dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE, 
                    print_summary=False):
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
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layers[-1])
    model5 = Model(transfer_model.input, output_layer)
    model5.compile(loss=losses.BinaryCrossentropy(), 
                   optimizer=optimizers.Adam(learning_rate=learning_rate), 
                   metrics=METRICS)
    if print_summary:
        model5.summary()
    return model5

def print_results(y_test, predictions):
    """
    Computes and prints accuracy, precision, recall, f1 score and confusion matrix
    """
    y_pred = np.round(np.squeeze(predictions))
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    #print(str(accuracy) + " | " + str(precision) + " | " + str(recall) + " | " + str(f1) + " | " + str(conf_mat) + " |")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1)
    print("Confusion Matrix: ", conf_mat)
    return accuracy, precision, recall, f1, conf_mat

def plot_results(acc_list, prec_list, rec_list, f1_list, model_names):
    """
    Plots accuracy, precision, recall and f1 score for each model
    """
    plt.figure(figsize=(20,10))
    plt.plot(acc_list, 'o', label="Accuracy")
    plt.plot(prec_list, 'o', label="Precision")
    plt.plot(rec_list, 'o', label="Recall")
    plt.plot(f1_list, 'o', label="F1 Score")
    plt.title("Project 5 Results")
    plt.xticks(range(len(model_names)),model_names, rotation=45)
    plt.xlabel("Model")
    plt.legend(loc='best')
    plt.show() 

def plot_roc_curve(y_test, pred_lst, labels_lst):
    """
    Builds and plots roc curve for given models
    """
    plt.figure(figsize=(20,10))
    for model_num in range(len(pred_lst)):
        y_pred = np.round(np.squeeze(pred_lst[model_num]))
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label=labels_lst[model_num])
    plt.title('Project 5 ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()
        
        
    