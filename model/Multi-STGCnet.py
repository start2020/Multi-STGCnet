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

path = "./trend.xlsx"
out_trend = pd.read_excel(path)[['labels','samples']]
out_trend_standard = out_trend / 9900

#################### hyperparameter
intervals = out.shape[0] 		 # 6640 days															# 12intervals
begin = 13 # start from 14th
interval_batch=35 # batch_size
n = 118                                  # number of stations

# tr
pre_trend = 1 
pre_periodic = 7 
pre_recent = 12 

# sr
pre_sr = 12 

#################### samples and labels
labels = []

samples_lstm_recent = []
samples_lstm_periodic = []
samples_lstm_trend = []
samples_sr = []

for i in range(begin*interval_batch,intervals):
    # labels
    label = out_standard['坪洲'].values[i]
    labels.append(label)
    # lstm_trend
    sample_lstm_trend = out_trend_standard['samples'].values[i]
    samples_lstm_trend.append(sample_lstm_trend)
    # lstm_periodic
    sample_lstm_periodic = out_standard['坪洲'].values[[i-35*7,i-35*6,i-35*5,i-35*4,i-35*3,i-35*2,i-35*1]]
    samples_lstm_periodic.append(sample_lstm_periodic)
    # lstm_recent
    sample_lstm_recent = out_standard['坪洲'].values[i-pre_recent:i]
    samples_lstm_recent.append(sample_lstm_recent)
    # sr
    sample_sr = out_standard.values[i-pre_sr:i]
    samples_sr.append(sample_sr)    
    
num_samples = len(labels)
labels = np.array(labels)

samples_sr = np.reshape(np.array(samples_sr),(num_samples,118,pre_sr))
samples_lstm_trend = np.reshape(np.array(samples_lstm_trend),(num_samples,pre_trend,1))
samples_lstm_periodic = np.reshape(np.array(samples_lstm_periodic),(num_samples,pre_periodic,1))
samples_lstm_recent = np.reshape(np.array(samples_lstm_recent),(num_samples,pre_recent,1))

print(labels.shape)
print(samples_sr.shape)
print(samples_lstm_trend.shape)
print(samples_lstm_periodic.shape)
print(samples_lstm_recent.shape)

#################### adjacency matrix
path ="./near matrix.xlsx"
dataset= pd.read_excel(path,index_col=0).values
adjacencys_near = []
for i in range(num_samples):
    adjacencys_near.append(dataset)
adjacencys_near = np.array(adjacencys_near)
print(adjacencys_near.shape)

path ="./middle matrix.xlsx"
dataset= pd.read_excel(path,index_col=0).values
adjacencys_middle = []
for i in range(num_samples):
    adjacencys_middle.append(dataset)
adjacencys_middle = np.array(adjacencys_middle)
print(adjacencys_middle.shape)

path ="./distant matrix.xlsx"
dataset= pd.read_excel(path,index_col=0).values
adjacencys_distant = []
for i in range(num_samples):
    adjacencys_distant.append(dataset)
adjacencys_distant = np.array(adjacencys_distant)
print(adjacencys_distant.shape)


#################### train and test split
x_train_sr = samples_sr[:4000]
adjacency_train_near = adjacencys_near[:4000]
adjacency_train_middle = adjacencys_middle[:4000]
adjacency_train_distant = adjacencys_distant[:4000]

x_train_lstm_recent = samples_lstm_recent[:4000]
x_train_lstm_periodic = samples_lstm_periodic[:4000]
x_train_lstm_trend = samples_lstm_trend[:4000]

x_test_sr = samples_sr[4000:]
adjacency_test_near = adjacencys_near[4000:]
adjacency_test_middle = adjacencys_middle[4000:]
adjacency_test_distant = adjacencys_distant[4000:]

x_test_lstm_recent = samples_lstm_recent[4000:]
x_test_lstm_periodic = samples_lstm_periodic[4000:]
x_test_lstm_trend = samples_lstm_trend[4000:]

y_train = labels[:4000]
y_test = labels[4000:]

print(x_train_sr.shape,adjacency_train_near.shape,adjacency_train_middle.shape,adjacency_train_distant.shape)
print(x_train_lstm_recent.shape,x_train_lstm_periodic.shape,x_train_lstm_trend.shape)

print(x_test_sr.shape,adjacency_test_near.shape,adjacency_test_middle.shape,adjacency_test_distant.shape)
print(x_test_lstm_recent.shape,x_test_lstm_periodic.shape,x_test_lstm_trend.shape)

print(y_train.shape,y_test.shape)


################################################ Model: Multi-STGCnet-SR
# input
features = Input(shape=(n,pre_sr))
adjacency_near = Input(shape=(n,n))
adjacency_middle = Input(shape=(n,n))
adjacency_distant = Input(shape=(n,n))

# near
# GCN layer
output_near_start = layers.Dot(axes=1)([adjacency_near, features])
output = layers.Dense(n,activation='relu')(output_near_start)
# GCN layer
output = layers.Dot(axes=1)([adjacency_near, output])
output = layers.Dense(n,activation='relu')(output)
output = layers.Permute((2,1))(output)
output = layers.LSTM(32,return_sequences=True)(output)
output = layers.LSTM(12,kernel_initializer='random_normal')(output)
output_near_end = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(output)


# middle
# GCN layer
output_middle_start = layers.Dot(axes=1)([adjacency_middle, features])
output = layers.Dense(n,activation='relu')(output_middle_start)
# GCN layer
output = layers.Dot(axes=1)([adjacency_middle, output])
output = layers.Dense(n,activation='relu')(output)
output = layers.Permute((2,1))(output)
output = layers.LSTM(32,return_sequences=True)(output)
output = layers.LSTM(12,kernel_initializer='random_normal')(output)
output_middle_end = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(output)

# distant
# GCN layer
output_distant_start = layers.Dot(axes=1)([adjacency_distant, features])
output = layers.Dense(n,activation='relu')(output_distant_start)
# GCN layer
output = layers.Dot(axes=1)([adjacency_distant, output])
output = layers.Dense(n,activation='relu')(output)
output = layers.Permute((2,1))(output)
output = layers.LSTM(32,return_sequences=True)(output)
output = layers.LSTM(12,kernel_initializer='random_normal')(output)
output_distant_end = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(output)

# merge
merge = layers.concatenate([output_near_end, output_middle_end ,output_distant_end],axis=-1)
temp = layers.Dense(35,activation='relu',kernel_initializer='random_normal')(merge)
output_end_SR = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(temp)


# model
model = models.Model(inputs=[features,adjacency_near,adjacency_middle,adjacency_distant],outputs=[output_end])
model.summary()
model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse','mape'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='SR.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)
                 ]

################################################ Model training
epochs = 1000
H = model.fit([x_train_sr,adjacency_train_near,adjacency_train_middle,adjacency_train_distant], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=([x_test_sr,adjacency_test_near,adjacency_test_middle,adjacency_test_distant],y_test))

################################################ Loss 
train_loss = H.history['loss']
test_loss = H.history['val_loss']
iterations = [i for i in range(epochs)]
plt.plot(iterations, train_loss,'b-',label='train_mae')
plt.plot(iterations, test_loss,'r-',label='test_mae')
plt.legend()
plt.title('Train_mae VS Test_mae')
################################################ predict
path="./SR.h5"
model_best = models.load_model(path)
predict_best = model_best.predict([x_test_sr,adjacency_test_near,adjacency_test_middle,adjacency_test_distant])
plt.figure(figsize=(20,10))
plt.plot(y_test,'r')
plt.plot(predict_best,'g')
