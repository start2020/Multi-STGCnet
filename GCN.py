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
pre_sr = 12 															# 12intervals
begin = 13 # start from 14th
interval_batch=35 # batch_size
n = 118                                  # number of stations

#################### samples and labels
labels = []
samples_near = []
for i in range(begin*interval_batch,intervals):
    # 标签
    label = out_standard['坪洲'].values[i]
    # lstm的样本
    sample_near = out_standard.values[i-pre_sr:i]
    samples_near.append(sample_near)
    labels.append(label)

num_samples = len(labels)
samples_near = np.reshape(np.array(samples_near),(num_samples,118,pre_sr))
labels = np.array(labels)
print(samples_near.shape)
print(labels.shape)
############################ adjacecny matrix
path ="./adjacecny matrix.xlsx"
dataset = pd.read_excel(path,index_col=0).values
adjacencys = []
for i in range(num_samples):
    adjacencys.append(dataset)
adjacencys = np.array(adjacencys)
print(adjacencys.shape)

#################### train and test split
x_train_near = samples_near[:4000]
x_test_near = samples_near[4000:]
adjacency_train_near = adjacencys[:4000]
y_train = labels[:4000]
y_test = labels[4000:]
adjacency_test_near = adjacencys[4000:]

print(x_train_near.shape,y_train.shape,adjacency_train_near.shape)
print(x_test_near.shape,y_test.shape,adjacency_test_near.shape)
plt.figure(figsize=(20,10))
plt.plot(y_test,'r')

################################################ Model:GCN
from keras import Input, models, layers
n = 118
input0 = Input(shape=(n,pre_sr))
adjacency = Input(shape=(n,n))

input1 = layers.Dot(axes=1)([adjacency, input0])
input2 = layers.Dense(n,activation='relu')(input1)

input3 = layers.Dot(axes=1)([adjacency, input2])
input4 = layers.Dense(n,activation='relu')(input3)

# merge
input5 = layers.Dense(1,activation='relu')(input4)
input6 = layers.Flatten()(input5)
output = layers.Dense(1,activation='relu')(input6)

model = models.Model(inputs=[input0,adjacency],outputs=[output])
model.summary()
model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse','mape'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='GCN.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)
                 ]

################################################ Model training
epochs = 1000
H = model.fit([x_train_near,adjacency_train_near], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=([x_test_near,adjacency_test_near],y_test))

################################################ Loss 
train_loss = H.history['loss']
test_loss = H.history['val_loss']
iterations = [i for i in range(epochs)]
plt.plot(iterations, train_loss,'b-',label='train_mae')
plt.plot(iterations, test_loss,'r-',label='test_mae')
plt.legend()
plt.title('Train_mae VS Test_mae')

################################################ predict
path="./GCN.h5"
model_best = models.load_model(path)
model.summary()
predict_best = model_best.predict([x_test_near,adjacency_test_near])
