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


#读取一个名为 icml_face_data.csv 文件
data = pd.read_csv('../input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
data.columns = ['emotion', 'Usage', 'pixels']  #将 DataFrame 中列的名称更改为指定的名称。在这里，对 data DataFrame 中的列名称进行更改，将 emotion, Usage, pixels 分别赋值给三列的列名。
#icml_face_data.csv 文件包含三列数据：情感标签（emotion）、数据用途（Usage）和图像像素信息（pixels）。
#其中 emotion 列包含七种可能的情感标签：愤怒、厌恶、恐惧、开心、哀伤、惊讶和中性。Usage 列指示了该图像是用于训练、测试还是验证。pixels 列包含一串用逗号分隔的像素值，需要进行解码才能获取原始图像。


#查看 DataFrame 的前几行数据，默认情况下，head() 方法会显示前五行数据。这可以帮助我们对数据集的结构和内容有一个初步的了解，并检查数据是否已经成功导入。
data.head()

#计算每个类别的数据个数。在这里，我们使用了该方法来计算 Usage 列中每个类别（Training、PublicTest 和 PrivateTest）的数据个数。
data.Usage.value_counts()
#value_counts() 函数返回一个 Series，其中包含每个唯一值的出现次数。该 Series 按值计数降序排列，并且具有唯一值的客户端将在左侧显示，其相应计数值将在右侧显示。

#定义函数，从输入的 data 数据中准备出用于训练的图像数据和对应的标签数据。
def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48)) 
        image_array[i, :, :, 0] = image / 255

    return image_array, image_label
#首先定义了一个 image_array 的 numpy 数组，用于存储所有的图像数据。image_array 的大小为 (len(data), 48, 48, 1)，表示有 len(data) 张图片，每张图片的大小为 48x48，且只有一个“通道”（即黑白图像）。
#接着定义了一个 image_label 数组，用于存储标签信息。image_label 包含.data 中每个样本的情感标签（emotion），需要将其转换为整数数组，使用 map(int, data['emotion']) 将每个情感标签转换为整数，并将其作为数组存储在 image_label 中。
#使用 for 循环，遍历 data 中的每一行，将每行的图像像素信息转换为 48x48 灰度图像，并将其存储在 image_array 对应位置里。
#首先使用 np.fromstring 展开 data DataFrame 中表示图像像素的字符串数据，并使用 int 类型存储每个像素值。然后使用 np.reshape 将展开的数组形状变为 48x48，最后将 48x48 的图像数据除以 255 进行归一化，以便于神经网络的训练。
#最终将准备好的图像数据 image_array 和标签数据 image_label 作为函数的返回值。



#自定义函数，在图像上展示给定 label 对应的不同表情数据。
def plot_examples(label):                     #在这个函数中，我们有一个参数 label，该参数表示我们要展示的表情类型。
    fig, axs = plt.subplots(1, 5, figsize=(25, 12))
    fig.subplots_adjust(hspace = .2, wspace=.2)
    axs = axs.ravel()
    for i in range(5):
        idx = data[data['emotion']==label].index[i]
        axs[i].imshow(train_images[idx][:,:,0], cmap='gray')
        axs[i].set_title(emotions[label])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
#首先创建一个包含 5 个子图的图形，并将其中的每个子图分配给 axs 数组。
#然后遍历这 5 个子图，找到在 data 中第一个符合给定标签 label 的图像，将其绘制到当前子图中。
#其中，idx 记录了这个图像的在数据中的索引，train_images 则是在其它地方定义的一个存放所有图片数据的 numpy 数组。
#接着，我们把展示的每个子图的 title 设置为 emotions[label]，emotions 是一个 python 字典类型，包含了每种可能的表情标签的名称。
#最后，我们将所有坐标标签设置为 []，以隐藏其在图像旁边。




#在图像上展示所有可能的表情类型。
def plot_all_emotions():
    N_train = train_labels.shape[0]

    sel = np.random.choice(range(N_train), replace=False, size=16)

    X_sel = train_images[sel, :, :, :]         #train_images 包含所有训练用图像数据
    y_sel = train_labels[sel]                 #train_labels 包含训练用所有数据的标签。

    plt.figure(figsize=[12,12])
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(X_sel[i,:,:,0], cmap='binary_r')
        plt.title(emotions[y_sel[i]])
        plt.axis('off')
    plt.show()
#首先从训练集中随机选择 16 个样本，分别用 X_sel 和 y_sel 存储像素数据和对应标签。
#然后，我们将像素数据展示出来，其中第 i 个子图的像素信息来自 X_sel 中的第 i 个索引出的图像。
#我们将像素信息展示为黑白图像，并在其下方显示相应的表情标签，以便于人类观测和理解。



#展示测试集的一张图像及其正确标签和模型的预测标签.
def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):
    """ Function to plot the image and compare the prediction results with the label """
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    
    bar_label = emotions.values()
    
    axs[0].imshow(test_image_array[image_number], 'gray')
    axs[0].set_title(emotions[test_image_label[image_number]])
    
    axs[1].bar(bar_label, pred_test_labels[image_number], color='orange', alpha=0.7)
    axs[1].grid()
    
    plt.show()
#首先定义一个包含两个子图的 figure，并将图像、正确标签和预测标签分别展示在它的不同子图中。
#在子图1中，我们将测试集（即 test_image_array）中的某个图像信息展示为灰度图像，并在其标题中展示相应的情感 test_image_label。
#在子图2中，我们用橙色的条形图展示预测标签 pred_test_labels，x轴方向为所有可能情感标签，y轴方向为每个标签对应的概率值。
#同时，我们添加了一个网格线以帮助更好地查看条形图中的信息。
#通过这个函数，我们可以方便地比较模型的预测结果和真实的情感标签，以寻找可能需要改进的方向来提高模型准确率。 



#在训练过程中可视化模型的 loss 和 accuracy 情况.
def vis_training(hlist, start=1):  
    
    loss = np.concatenate([h.history['loss'] for h in hlist])
    val_loss = np.concatenate([h.history['val_loss'] for h in hlist])
    acc = np.concatenate([h.history['accuracy'] for h in hlist])
    val_acc = np.concatenate([h.history['val_accuracy'] for h in hlist])
    
    epoch_range = range(1,len(loss)+1)

    plt.figure(figsize=[12,6])
    plt.subplot(1,2,1)
    plt.plot(epoch_range[start-1:], loss[start-1:], label='Training Loss')
    plt.plot(epoch_range[start-1:], val_loss[start-1:], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epoch_range[start-1:], acc[start-1:], label='Training Accuracy')
    plt.plot(epoch_range[start-1:], val_acc[start-1:], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.show()
#该函数依赖于一个包含历史记录的列表 hlist，这将是一个 Keras 训练历史对象列表，每个历史对象都包含有模型在训练期间保存的各种度量指标（如 loss 和 accuracy）。
#首先将所有历史记录汇总到一个新的 numpy 数组中。然后，我们定义了一个 epoch_range 用来表示训练记录的轮数。
#接着，我们创建一个 1 x 2 的 subplots 图表，用于展示训练和验证 loss 和 accuracy 的情况。
#在左侧，我们展示了 loss 的变化过程（训练 loss 和验证 loss）。
#在右侧，我们展示了 accuracy 的变化过程（训练 accuracy 和验证 accuracy）。在图像中，x轴表示 epoch 的数量，y轴代表相应度量的值。


#定义了一个名为 emotions 的 Python 字典对象，它包含了 FER2013 数据集中的七个情感标签及其对应的名称。
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
#这个字典可以方便地将数字的情感标签转换为对应的文本标签，以便于人类的理解和解释。

#数据准备，对数据集进行预处理,划分训练集和测试集
full_train_images, full_train_labels = prepare_data(data[data['Usage']=='Training'])    #data是输入的原始数据
test_images, test_labels = prepare_data(data[data['Usage']!='Training'])
#使用了一个名为prepare_data的函数对数据进行处理。在这里，数据集被分为训练集和测试集，其中训练集包含所有标记为"Training"的数据，测试集则包含其他用途的数据。
#最终，full_train_images和full_train_labels变量包含了所有训练集的图像和标签数据，test_images和test_labels则包含了所有测试集的图像和标签数据。



print(full_train_images.shape)
print(full_train_labels.shape)
print(test_images.shape)
print(test_labels.shape)



#划分训练集和验证集
train_images, valid_images, train_labels, valid_labels =\
    train_test_split(full_train_images, full_train_labels, test_size=0.2, random_state=1)
#用scikit-learn库中的train_test_split函数，将full_train_images和full_train_labels分割成训练集（train_images和train_labels）和验证集（valid_images和valid_labels）。
#其中test_size=0.2表示将数据集按照80%和20%的比例分割到训练集和验证集中。同时指定random_state=1，以便在多次运行代码时得到同样的结果。


print(train_images.shape)
print(valid_images.shape)
print(train_labels.shape)
print(valid_labels.shape)


plot_all_emotions()
plot_examples(label=0)
plot_examples(label=1)
plot_examples(label=2)
plot_examples(label=3)
plot_examples(label=4)
plot_examples(label=5)
plot_examples(label=6)



#计算并设置分类模型中不同情绪类别的权重
class_weight = dict(zip(range(0, 7), (((data[data['Usage']=='Training']['emotion'].value_counts()).sort_index())/len(data[data['Usage']=='Training']['emotion'])).tolist()))
#首先从原始的数据集中选择出标记为"Training"的所有数据，并统计每种情绪在训练集中的数量（即情绪类别的频率）。
#然后，使用Python内置函数range()和zip()将情绪类别的数字编号（0到6）和它们的频率配对组成字典，方便后续在模型中使用。并将这种情绪类别权重字典赋值给class_weight变量，用于训练分类器时调整不同情绪类别的权重。
#这个过程的目的是为了平衡不同情绪类别在数据集中出现的次数，以避免一些情绪类别的样本数量过少而导致模型训练效果不佳。

class_weight


tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect() #使用Google Colab平台上的TPU（tensor processing unit）加速训练CNN模型。
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    model = Sequential()     #首先，使用Sequential()函数创建一个顺序模型

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(48,48,1)))
#添加了一个包含128个卷积核的卷积层（即Conv2D层），该层使用‘relu’激活函数和‘same’填充，输入形状为（48，48，1），网络采用3×3的卷积核进行卷积操作来提取特征。same填充使得卷积输出与输入的尺寸相等。    

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
#添加了另一个卷积层，与上面的卷积层相同，形成双重卷积操作，
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#紧跟着一个最大池化层，尺寸为2×2且步长为2，它通过将特征图的大小折半来减少计算量。
    model.add(Dropout(0.25))
#同时还添加了25%的Dropout层，防止过拟合。
    model.add(BatchNormalization())
#然后，添加了BatchNormalization层，将卷积层或全连接层的输出归一化，使之规范化, 加速学习过程.


#重复上述操作，添加特征数量加倍的卷积层和降采样池化层，中间还加入BatchNormalization层和Dropout层。
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
#最后，将输出的张量尺寸从高维展平，将其展平为一维张量。
    model.add(Flatten())
#添加三个深度为512的全连接层，并在每个层之间使用Dropout层防止过拟合。
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
#最后一个全连接层用softmax作为激活函数，将预测转化为概率。
    model.add(Dense(7, activation='softmax'))



#设置Adam优化器对模型进行编译
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy'])
#学习率为0.001, 然后对模型进行编译，选择’categorical_crossentropy’作为损失函数进行多元分类。
#metrics是准确率作为性能评估指标，经过优化和改进后就构建出了一个可以用于情绪识别的深度卷积神经网络模型。

#打印网络模型结构
model.summary()
#在调用model.summary()后，会输出模型的层（Layer）、输出形状（Output Shape）、可训练参数数量（Param #）等信息。
#这个方法可以帮助我们确认当前模型的结构是否符合预期，以便调整和优化时对网络结构进行更精细的调整。
#同时，该方法也便于我们了解模型的参数量大小，帮助我们评估模型的计算资源要求和存储空间需求。


#使用fit()函数训练模型
h1 = model.fit(train_images, train_labels, batch_size=256, epochs=30, verbose=1, validation_data=(valid_images, valid_labels))
#使用了训练集的图片数据train_images和标签数据train_labels来拟合CNN模型，指定了每次训练的图片数量batch_size为256，训练次数为30次epochs。
#同时，verbose=1表示将训练过程中的输出显示为进度条。
#此外，使用了验证集数据（valid_images和valid_labels）来检查模型的性能，并进行模型的评估。验证数据可以有效地避免模型过度拟合训练数据。
#最终，将返回历史训练结果（h1），包括训练过程中每一轮的损失和准确率的变化情况，以及验证集上的损失和准确率评估结果。


#可视化训练过程
vis_training([h1])
#这是一个用于可视化模型训练过程的函数。函数接收一个历史记录（history）的列表，每个历史记录包含了训练模型过程中的准确性和损失数据。
#此函数将训练准确性、训练损失、验证准确性和验证损失输出到一组图表中，便于可视化每一轮训练的效果和趋势。


#调整优化器学习率
keras.backend.set_value(model.optimizer.learning_rate, 0.00001)
#使用了Keras的backend模块中的set_value()函数，将模型当前的学习率（learning_rate）设为0.00001。
#这是调整优化器学习率的一种方法，可以通过该方法来降低学习率进而提高训练的精度，有助于避免因为学习率过高造成的训练震荡不收敛的情况。

#重新进行模型训练
h2 = model.fit(train_images, train_labels, batch_size=256, epochs=30, verbose=1, 
                   validation_data =(valid_images, valid_labels)) 
#使用model.fit()函数重新进行模型训练，仍然采用batch_size为256、训练次数为30次epochs的训练方式，指定verbose=1让训练过程可以在终端中看到详细输出。
#此时，我们为了避免模型过度拟合训练数据，使用验证数据（valid_images和valid_labels）进行模型性能的评估，验证的结果将输出到终端中，并且也可以用vis_training()方法进行可视化呈现历史记录（history）的变化情况。

vis_training([h1, h2])
#该函数传递了两个历史记录（h1和h2）作为参数，分别代表了模型重新训练之前和之后的历史训练结果。
#该函数将训练准确性、训练损失、验证准确性和验证损失输出到一组图表中，方便我们直观地了解多种模型的训练效果和性能，并进行比较分析。

#使用训练完成的CNN模型对测试集（test_images和test_labels）进行预测，并输出该模型的测试准确率。
test_prob = model.predict(test_images) #首先，使用model.predict()函数对测试集的图片数据进行预测，得到每个类别的预测得分，这里将结果保存为test_prob。
test_pred = np.argmax(test_prob, axis=1) #接着，使用NumPy库中的argmax()函数找到每个样本得分最高的类别作为该样本的预测结果，将结果保存为test_pred。
test_accuracy = np.mean(test_pred == test_labels) #最后，通过计算预测结果和实际标签相同的样本所占的比例，得到模型的测试准确率并将其保存为test_accuracy，输出在终端上展示。

print(test_accuracy)

#计算测试数据(test_data)上情感分类器的混淆矩阵，并将结果以DataFrame形式输出。
conf_mat = confusion_matrix(test_labels, test_pred)
#confusion_matrix(test_labels, test_pred)用来计算测试标签(test_labels)和预测标签(test_pred)之间的混淆矩阵。
#混淆矩阵是一个二维数组，其中行表示真实标签，列表示预测标签，对角线上的元素表示正确分类的数量，非对角线上的元素代表错误分类的数量。

pd.DataFrame(conf_mat, columns=emotions.values(), index=emotions.values())
#使用pd.DataFrame(conf_mat, columns=emotions.values(), index=emotions.values())创建一个名为conf_mat的DataFrame对象。
#第一个参数conf_mat是之前计算得到的混淆矩阵，第二个和第三个参数分别是列索引和行索引，它们都使用情感类型的列表emotions.values()。

#通过这个DataFrame对象，我们可以更方便地分析情感分类器在每个类别上的性能表现，例如哪些类别常常被混淆，哪些类别分类效果最好等等。

#绘制混淆矩阵的热力图，并显示出来。
fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_normed=True,
                                show_absolute=False,
                                class_names=emotions.values(),
                                figsize=(8, 8))
fig.show()


#用来输出情感分类器在测试数据上的分类性能报告，包括准确率、召回率、F1分数等指标。
print(classification_report(test_labels, test_pred, target_names=emotions.values()))
#第一个参数是测试标签(test_labels)，第二个参数是预测标签(test_pred)，第三个参数target_names是情感类型的列表。
#该函数会根据测试标签和预测标签计算出各种分类评价指标，包括精度、召回率、F1分数及它们的加权平均等等，并将结果以文本格式输出。


#将情感分类模型(model)保存为JSON格式的文件
model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
    
    
#将情感分类模型(model)保存为Keras模型文件
model.save('final_model.h5')
   
