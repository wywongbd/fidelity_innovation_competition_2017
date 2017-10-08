"""
Helper functions to visualize CNN implemented.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import sqrt, ceil

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
import h5py    # For loading keras model


def show_net_weights(W):
    """
    Visualize all the weights in grid form
    """
    W = W.reshape(2, 2, 2, -1).transpose(3, 0, 1, 2)
    # Transform to (_, _, _, 4) and take 3 of them
    W = np.repeat(W, 2, axis=3)
    W = W[:, :, :, 1:4]

    # Get the corresponding grid
    grid = visualize_grid(W, padding=1).astype('uint8')
    
    # Initialize subplot
    fig, ax = plt.subplots()
    img = ax.imshow(grid)
    plt.gca().axis('off')  # Turn off the value of axis
    ax.set_title('Visualization of fully connected layer')

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(img)
    # Vertically oriented colorbar
    cbar.ax.set_yticklabels(['-1', '-0.6', '-0.2', '0.2', '0.6', '1'])  

    plt.show()

def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(0, grid_size):
        x0, x1 = 0, W
        for x in range(0, grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
        # grid_max = np.max(grid)
        # grid_min = np.min(grid)
        # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

def train():
    model = create_model()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train model and return training history


    checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
    history = model.fit(X_train, y_train, batch_size=30, epochs=300, verbose=1,
                        validation_data=(X_test, y_test), callbacks=[checkpointer])

    model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True, validation_split=0.2, verbose=2, )

def load_trained_model(weights_path):
    model = create_model()
    model.load_weights(weights_path)
    
    return model

def create_model():
    """ Create model to be loaded by load_trained_model """
    # Hardcode some values
    cnn_input_shape = (15, 5, 1)
    num_classes = 3

   # Initialize model
    model = Sequential()

    # Layer1 (Convolutional and pooling layer)
    model.add(Conv2D(8, (2, 2), strides=1, activation='relu', input_shape=cnn_input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 2 (Convolutional and pooling layer)
    model.add(Conv2D(8, (2, 2), strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 3 (Fully connected layer)
    model.add(Flatten())  # Convert 3D matricx to 1D vector before going in fully connected layers
    model.add(Dense(4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 4 (Softmax layer)
    model.add(Dense(num_classes, activation='softmax'))

    return model

def plot_confusion_matrix(conf_arr):
    """
    Input:
    - conf_arr is 2D array for confusion matrix
    """
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(0, width):
        for y in range(0, height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = '012'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])

    plt.show()
   
def plot_learning_curve(history):
    """
    Plot the train/validation loss and accuracies
    Input:
    - history is dict returned by keras model.fit()
    """
    # Plot history for accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Plot history for loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
