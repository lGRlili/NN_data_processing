# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:56:35 2018

@author: zuodeng
"""
import multiprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import yaml
from keras import Sequential
from keras.layers import *
from keras.models import model_from_yaml
from keras.optimizers import *
from scipy import stats
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

vocab_dim = 300#词向量维度
n_exposures = 10#10
window_size = 7
cpu_count = multiprocessing.cpu_count()
n_iterations = 1
max_len = 10
input_length = 100
batch_size = 50
n_epoch = 2



# 划分测试集和训练集
train_data = pd.read_pickle('yunyingshang_data.pkl')
train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
print(train_data)
train_label = train_data['label']

test_data = train_data[32000:]
train_data = train_data[:32000]

test_label = test_data['label']


test_label = train_label[32000:]
train_label = train_label[:32000]

print(train_data.shape)
print(test_data.shape)
print(train_label.shape)
print(test_label.shape)


data_size = train_data.shape[0]


train_data.drop(['label','x_1','x_2'],axis = 1, inplace = True)
test_data.drop(['label','x_1','x_2'],axis = 1, inplace = True)

print(train_data.shape)
print(test_data.shape)
print(train_label.shape)
print(test_label.shape)
print('---'*45)
print(train_data)
print('---'*45)
X_train = train_data#np.array(list(train_data))

y_train = train_label#np.array(train_label)

X_test = test_data#np.array(list(test_data))

y_test = test_label#np.array(test_label)

# X_train=list(X_train)
# X_test=list(X_test)
print(X_train)
# print(y_train.shape)
# print(X_test)
# print(y_test.shape)


X_train = np.array(X_train).reshape(-1,1, 36, 20)

y_train = np.array(y_train).reshape(32000, 1)

X_test = np.array(X_test).reshape(-1, 1,36, 20)

y_test = np.array(y_test).reshape(8068,1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print('---'*45)
print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

Dropout_para = 0.3
#--------------------------------------------------------------------
# Another way to build your CNN
model = Sequential()
#--------------------------------------------------------------------
# Conv layer 1 output shape (32, 100, 8)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 36,20),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
     kernel_regularizer = regularizers.l2(0.15),
     activity_regularizer = regularizers.l1(0.0001)
))
model.add(Activation('relu'))
model.add(Dropout(Dropout_para))


#--------------------------------------------------------------------
# Pooling layer 1 (max pooling) output shape (32, 50, 4)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first'
))
model.add(Dropout(Dropout_para))
# #--------------------------------------------------------------------
# # Conv layer 2 output shape (64, 50, 4)
# model.add(Convolution2D(filters=64,
#                         kernel_size=5,
#                         strides=1,
#                         padding='same',
#                         data_format='channels_first'
#                        ))
# model.add(Activation('relu'))
# model.add(Dropout(Dropout_para))
# #--------------------------------------------------------------------
# # Pooling layer 2 (max pooling) output shape (64, 25, 2)
# model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
# model.add(Dropout(Dropout_para))
#--------------------------------------------------------------------
# Fully connected layer 1 input shape (64 * 25 * 2) = (), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(Dropout_para))
#--------------------------------------------------------------------
# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(1))
model.add(Activation('sigmoid'))
#--------------------------------------------------------------------
# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()








model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epoch)#,  validation_data=(X_test, y_test)

#plot_model(model, to_file='model.png')
print("Evaluate...")

loss, accuracy = model.evaluate(X_train, y_train,
                       batch_size=batch_size)

print('\ntrain loss: ', loss)
print('\ntrain accuracy: ', accuracy)

from sklearn import metrics
y_pred = model.predict(X_train)
print(y_pred)
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred, pos_label=1)
metrics.auc(fpr, tpr)
print('\ntrain AUC: ', metrics.auc(fpr, tpr))


plt.title('Receiver Operating Characteristic DNN')
plt.plot(fpr, tpr, 'b',  label='AUC = %0.2f'% metrics.auc(fpr, tpr))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

loss, accuracy = model.evaluate(X_test, y_test,
                       batch_size=batch_size)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

from sklearn import metrics
y_pred = model.predict(X_test)
print(y_pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
metrics.auc(fpr, tpr)
print('\ntest AUC: ', metrics.auc(fpr, tpr))

plt.title('Receiver Operating Characteristic DNN')
plt.plot(fpr, tpr, 'b',  label='AUC = %0.2f'% metrics.auc(fpr, tpr))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

yaml_string = model.to_yaml()
with open('CNN.yml', 'w') as outfile:
    outfile.write(yaml.dump(yaml_string, default_flow_style=True))
model.save_weights('CNN.h5')


#
# #定义ks函数
# get_ks = lambda y_pred,y_true: stats.ks_2samp(y_pred[y_true==1][0], y_pred[y_true!=1][0]).statistic
#
# print(y_pred.shape)
# print(y_test.shape)
# print(y_pred)
# y_pred=(y_pred)
# print(y_pred)
# print(type(y_pred))
# print(y_test)
# ks_values = get_ks(y_pred, y_test)
#
# print('ks_values', ks_values)

