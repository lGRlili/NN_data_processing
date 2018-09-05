# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:56:35 2018

@author: zuodeng
"""
import multiprocessing
import matplotlib.pyplot as plt
import lightgbm
import xgboost
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
n_epoch = 1



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


train_data.drop(['label'],axis = 1, inplace = True)
test_data.drop(['label'],axis = 1, inplace = True)

print(train_data.shape)
print(test_data.shape)
print(train_label.shape)
print(test_label.shape)



X_train = train_data#np.array(list(train_data))

y_train = train_label#np.array(train_label)

X_test = test_data#np.array(list(test_data))

y_test = test_label#np.array(test_label)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Another way to build your CNN
model = Sequential()

model.add(Dense(256, input_dim=722, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(512,activation='relu')) #没有input 表示隐层神经元
model.add(Dropout(0.2))

model.add(Dense(128,activation='relu')) #没有input 表示隐层神经元
model.add(Dropout(0.2))


model.add(Dense(1,activation='sigmoid')) #输出1维，表示是输出层神经元
# model.add(Activation('sigmoid'))
print('Compiling the Model...')

adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epoch)#,  validation_data=(X_test, y_test))

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






loss, accuracy = model.evaluate(X_test, y_test,
                       batch_size=batch_size)


yaml_string = model.to_yaml()
with open('DNN.yml', 'w') as outfile:
    outfile.write(yaml.dump(yaml_string, default_flow_style=True))
model.save_weights('DNN.h5')

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)

print(y_pred)

from sklearn import metrics
y_pred = model.predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
metrics.auc(fpr, tpr)
print('\ntest AUC: ',metrics.auc(fpr, tpr))



plt.title('Receiver Operating Characteristic DNN')
plt.plot(fpr, tpr, 'b',  label='AUC = %0.2f'% metrics.auc(fpr, tpr))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()