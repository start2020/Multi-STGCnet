import numpy as np
import pandas as pd
from keras import models, layers, Input
import matplotlib.pyplot as plt
import keras
################################################ Data process

#################### standardization
path = "./raw data.xlsx"
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
pre_sr = 12                              # number of intervals

#################### samples and labels
labels = []
samples_near = []
for i in range(begin*interval_batch,intervals):
    label = out_standard['坪洲'].values[i]
    sample_near = out_standard.values[i-pre_sr:i]
    samples_near.append(sample_near)
    labels.append(label)

num_samples = len(labels)
samples_near = np.reshape(np.array(samples_near),(num_samples,118,pre_sr))
labels = np.array(labels)
print(samples_near.shape)
print(labels.shape)

#################### adjacency matrix
path ="./near matrix.xlsx"
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


################################################ Model: Near Block - Multi-STGCnet
from keras import Input, models, layers

features = Input(shape=(n,pre_sr))      
adjacency = Input(shape=(n,n))           # adjacency matrix

#################### spatial component
# GCN layer
output = layers.Dot(axes=1)([adjacency, features])
output = layers.Dense(n,activation='relu')(output)
# GCN layer
output = layers.Dot(axes=1)([adjacency, output])
output = layers.Dense(n,activation='relu')(output)

#################### temporal component
# LSTM
output = layers.Permute((2,1))(output)
output = layers.LSTM(32,return_sequences=True)(output)
output = layers.LSTM(12,kernel_initializer='random_normal')(output)
# output layer
output = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(output)

model = models.Model(inputs=[features,adjacency],outputs=[output])
model.summary()
model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse','mape'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='Near.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)
                 ]

################################################ Model training
epochs = 1000
H = model.fit([x_train_near,adjacency_train_near], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_split=0.2)


################################################ Loss 
train_loss = H.history['loss']
test_loss = H.history['val_loss']
iterations = [i for i in range(epochs)]
plt.plot(iterations, train_loss,'b-',label='train_mae')
plt.plot(iterations, test_loss,'r-',label='test_mae')
plt.legend()
plt.title('Train_mae VS Test_mae')

################################################ predict
path="./Near.h5"
model_best = models.load_model(path)
predict_best = model_best.predict([x_test_near,adjacency_test_near])

