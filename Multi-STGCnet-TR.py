import numpy as np
import pandas as pd
from keras import models, layers, Input
import matplotlib.pyplot as plt
import keras
################################################ Data process

#################### standardization
path = "./daily.xlsx"
out = pd.read_excel(path)[['labels','samples']]
out_standard = out / 9900

path = "./features.xlsx"
out = pd.read_excel(path,index_col=0)
out_standard = out / 9900

#################### hyperparameter
intervals = out.shape[0] 		 # 6640 days
begin = 13 # start from 14th
interval_batch=35 # batch_size
n = 118                                  # number of stations
pre_daily = 1 
pre_periodic = 7 
pre_recent = 12 


#################### samples and labels
labels = []
samples_lstm_recent = []
samples_lstm_periodic = []
samples_lstm_daily = []
for i in range(begin*interval_batch,intervals):
    label = out_daily_standard['labels'].values[i]
    labels.append(label)
    sample_lstm_daily = out_daily_standard['samples'].values[i]
    samples_lstm_daily.append(sample_lstm_daily)
    sample_lstm_periodic = out_standard['坪洲'].values[[i-35*7,i-35*6,i-35*5,i-35*4,i-35*3,i-35*2,i-35*1]]
    samples_lstm_periodic.append(sample_lstm_periodic)
    sample_lstm_recent = out_standard['坪洲'].values[i-pre_recent:i]
    samples_lstm_recent.append(sample_lstm_recent)
    
num_samples = len(labels)
labels = np.array(labels)
print(labels.shape)

samples_lstm_daily = np.reshape(np.array(samples_lstm_daily),(num_samples,pre_daily,1))
samples_lstm_periodic = np.reshape(np.array(samples_lstm_periodic),(num_samples,pre_periodic,1))
samples_lstm_recent = np.reshape(np.array(samples_lstm_recent),(num_samples,pre_recent,1))
print(samples_lstm_daily.shape)
print(samples_lstm_periodic.shape)
print(samples_lstm_recent.shape)

#################### train and test split
x_train_lstm_recent = samples_lstm_recent[:4000]
x_test_lstm_recent = samples_lstm_recent[4000:]
x_train_lstm_periodic = samples_lstm_periodic[:4000]
x_test_lstm_periodic = samples_lstm_periodic[4000:]
x_train_lstm_daily = samples_lstm_daily[:4000]
x_test_lstm_daily = samples_lstm_daily[4000:]


y_train = labels[:4000]
y_test = labels[4000:]

print(x_train_lstm_recent.shape,y_train.shape)
print(x_test_lstm_recent.shape,y_test.shape)
print(x_train_lstm_periodic.shape,y_train.shape)
print(x_test_lstm_periodic.shape,y_test.shape)
print(x_train_lstm_daily.shape,y_train.shape)
print(x_test_lstm_daily.shape,y_test.shape)

plt.figure(figsize=(20,10))
plt.plot(y_test,'r')

################################################ Model: Multi-STGCnet-TR
# recent
output_recent_start = Input(shape=(pre_recent,1), dtype='float')
output = layers.LSTM(35,return_sequences=True,kernel_initializer='random_normal')(output_recent_start)
output = layers.LSTM(12,kernel_initializer='random_normal')(output)
output_recent_end = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(output)

# periodic
output_periodic_start = Input(shape=(pre_periodic,1), dtype='float')
output = layers.LSTM(35,return_sequences=True,kernel_initializer='random_normal')(output_periodic_start )
output = layers.LSTM(12,kernel_initializer='random_normal')(output)
output_periodic_end = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(output)

# daily
output_daily_start = Input(shape=(pre_daily,1), dtype='float')
output = layers.LSTM(35,return_sequences=True,kernel_initializer='random_normal')(output_daily_start)
output = layers.LSTM(12,kernel_initializer='random_normal')(output)
output_daily_end= layers.Dense(1,activation='relu',kernel_initializer='random_normal')(output)

# merge
merge = layers.concatenate([output_recent_end, output_periodic_end,output_daily_end],axis=-1)
temp = layers.Dense(35,activation='relu',kernel_initializer='random_normal')(merge)
output = layers.Dense(1,activation='relu',kernel_initializer='random_normal')(temp)
model = models.Model(inputs=[output_recent_start,output_periodic_start,output_daily_start],outputs=[output])

model.summary()
model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse','mape'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='TR.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)
                 ]

################################################ Model training
epochs = 1000
H = model.fit([x_train_sr,adjacency_train_near,adjacency_train_middle,adjacency_train_distant,x_train_lstm_recent,x_train_lstm_periodic,x_train_lstm_daily], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=([x_test_sr,adjacency_test_near,adjacency_test_middle,adjacency_test_distant,x_test_lstm_recent,x_test_lstm_periodic,x_test_lstm_daily],y_test))
################################################ Loss 
train_loss = H.history['loss']
test_loss = H.history['val_loss']
iterations = [i for i in range(epochs)]
plt.plot(iterations, train_loss,'b-',label='train_mae')
plt.plot(iterations, test_loss,'r-',label='test_mae')
plt.legend()
plt.title('Train_mae VS Test_mae')

################################################ predict
path="./ALL.h5"
model_best = models.load_model(path)
predict_best = model_best.predict([x_test_sr,adjacency_test_near,adjacency_test_middle,adjacency_test_distant,x_test_lstm_recent,x_test_lstm_periodic,x_test_lstm_daily])

plt.figure(figsize=(20,10))
plt.plot(y_test,'r')
plt.plot(predict_best,'g')
