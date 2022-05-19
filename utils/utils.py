

import random

import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
import numpy as np
import pandas as pd
import tensorflow as tf
from .Config import Config
import random
import os
from matplotlib.ticker import FixedLocator, FixedFormatter













def plot_confusion_matrix(confusion_matrix, target_names, title='Confusion matrix', cmap=None, normalize=True, save_result = False):
    """
    source: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix

    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    confusion_matrix: confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    fig, ax = plt.subplots(1, 1, figsize = (3.3, 3.0))
    plt.subplots_adjust(left=0.15, bottom=0.14, right=0.96, top=0.90,
                wspace=None, hspace=None)
    text_size = 10
    
    accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix)) *100
    misclass = 100 - accuracy


    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.colorbar(fraction=0.045, pad=0.05)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names,  size=text_size)#rotation=45,
        plt.yticks(tick_marks, target_names, size=text_size)



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > thresh else "black"
        if i == j:
  
            color = 'white'

        if normalize:
            plt.text(
                j, i, "{:0.2f}".format(cm[i, j]*100),
                horizontalalignment="center",
                color=color,
                fontdict={'family' : 'Times New Roman', 'size'   : text_size}
            )
        else:
            plt.text(
                j, i, "{:,}".format(cm[i, j]*100),
                horizontalalignment="center",
                color=color,
                fontdict={'family' : 'Times New Roman', 'size'   : text_size}
            )

    plt.tight_layout()
    plt.ylabel('True label',fontdict={'family' : 'Times New Roman', 'size'   : text_size}, labelpad = 0.2)
    plt.xlabel('Predicted label',fontdict={'family' : 'Times New Roman', 'size'   : text_size})
   
    if save_result:
        filename = Config.RESULT_DIR + '/'+ 'picture/' + 'matrix.pdf'
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                           inter_op_parallelism_threads=1)


    tf.compat.v1.set_random_seed(seed)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def plot_results(history):
    """
    Saves as a .png image a graph of learning error
    :param history:
    :param name:
    """

    
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss', 'train se','test se', 'val ppv'], loc='upper left')

    plt.show()
    
    plt.figure()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'test acc', 'train se','test se', 'val ppv'], loc='upper left')

    plt.show()
    
    



def setup_gpu(number = '0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    