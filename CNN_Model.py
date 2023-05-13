import numpy as np 
import pandas as pd 
import os


#读取数据集
for dirname, _, filenames in os.walk('/kaggle/input'):         #os.walk 是 Python 自带的一个目录遍历函数，用于在一个目录树中游走输出文件名、文件夹名。
    for filename in filenames:
        print(os.path.join(dirname, filename))
#在 Kaggle 平台上，数据集都位于 /kaggle/input 文件夹下。这里通过遍历该路径下的所有文件和文件夹，并打印出并打印输出文件名和路径。

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

import keras
from keras.models import Sequential
from keras.layers import *
from keras_preprocessing.image import ImageDataGenerator

import zipfile 

import cv2
import seaborn as sns
%matplotlib inline

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix

from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
