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

#################### hyperparameter
intervals = out.shape[0] 									# 6640 days
pre = 1 															# 12intervals
begin = 13 # start from 14th
interval_batch=35 # batch_size
n = 118                                  # number of stations
pre_sr = 12                              # number of intervals

#################### samples and labels
labels = []
samples_lstm_daily = []
for i in range(begin*interval_batch,intervals):
    label = out_standard['labels'].values[i]
    sample_lstm_daily = out_standard['samples'].values[i]
    samples_lstm_daily.append(sample_lstm_daily)
    labels.append(label)

num_samples = len(labels)
samples_lstm_daily = np.reshape(np.array(samples_lstm_daily),(num_samples,pre,1))
labels = np.array(labels)
print(samples_lstm_daily.shape)
print(labels.shape)

#################### train and test split
x_train_lstm_daily = samples_lstm_daily[:4000]
x_test_lstm_daily = samples_lstm_daily[4000:]
y_train = labels[:4000]
y_test = labels[4000:]

print(x_train_lstm_daily.shape,y_train.shape)
print(x_test_lstm_daily.shape,y_test.shape)

plt.figure(figsize=(20,10))
plt.plot(y_test,'r')

################################################ Model: daily Block - Multi-STGCnet
input130 = Input(shape=(pre,1), dtype='float')
input131 = layers.LSTM(35,return_sequences=True)(input130)
input132 = layers.LSTM(12)(input131)
output13 = layers.Dense(1,activation='relu')(input132)
model = models.Model(inputs=[input130],outputs=[output13])
model.summary()
model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse','mape'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='lstm_daily.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)

################################################ Model training
epochs = 1000
H = model.fit([x_train_lstm_daily], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=([x_test_lstm_daily],y_test))

################################################ Loss 
train_loss = H.history['loss']
test_loss = H.history['val_loss']
iterations = [i for i in range(epochs)]
plt.plot(iterations, train_loss,'b-',label='train_mae')
plt.plot(iterations, test_loss,'r-',label='test_mae')
plt.legend()
plt.title('Train_mae VS Test_mae')

################################################ predict
path="./lstm_daily.h5"
model_best = models.load_model(path)
model_best.summary()
predict_best = model_best.predict(x_test_lstm_daily)
