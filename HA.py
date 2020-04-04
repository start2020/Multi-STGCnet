import numpy as np
import pandas as pd
from keras import models, layers, Input
import matplotlib.pyplot as plt
import keras
################################################ Data process

#################### standardization
path = "./features.xlsx"
out = pd.read_excel(path,index_col=0)
Max_out = np.max(out.values)
Min_out = np.min(out.values)
range_out = Max_out - Min_out
out_standard = out / range_out
print(Max_out,Min_out)

#################### hyperparameter
intervals = out.shape[0] 									# 6640 days
pre = 12 															# 12intervals
begin = 13 # start from 14th
interval_batch=35 # batch_size
n = 118                                  # number of stations

#################### samples and labels
labels = []
samples_lstm_recent = []
for i in range(begin*interval_batch,intervals):
    # labels
    label = out_standard['坪洲'].values[i]
    # lstm
    sample_lstm_recent = out_standard['坪洲'].values[i-pre:i]
    samples_lstm_recent.append(sample_lstm_recent)
    labels.append(label)

num_samples = len(labels)
samples_lstm_recent = np.array(samples_lstm_recent)
labels = np.array(labels)
print(samples_lstm_recent.shape)
print(labels.shape)

#################### train and test split
x_train_near = samples_near[:4000]
x_train_lstm_recent = samples_lstm_recent[:4000]
x_test_lstm_recent = samples_lstm_recent[4000:]
y_train = labels[:4000]
y_test = labels[4000:]

print(x_train_lstm_recent.shape,y_train.shape)
print(x_test_lstm_recent.shape,y_test.shape)
################################################ Model:HA
predicts = []
for i in range(y_test.shape[0]):
    predict = np.mean(x_test_lstm_recent[i])
    print(predict,y_test[i])
    predicts.append(predict)
predicts = np.array(predicts)

################################################ Predict
path = "./results.xlsx"
df = pd.read_excel(path)
df['HA'] = predicts*9900
df.to_excel(path,index = False)

plt.figure(figsize=(20,10))
plt.plot(y_test,'r')
plt.plot(predicts,'g')

