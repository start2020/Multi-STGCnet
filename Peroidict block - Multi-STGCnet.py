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
pre = 7 # use pre intervals to predict pre+1 interval 															# 12intervals
begin = 13 # start from 14th
interval_batch=35 # batch_size
n = 118                                  # number of stations

#################### samples and labels
labels = []
samples_lstm_periodic = []
for i in range(begin*interval_batch,intervals):
    label = out_standard['坪洲'].values[i]
    sample_lstm_periodic = out_standard['坪洲'].values[i-pre:i]
    samples_lstm_periodic.append(sample_lstm_periodic)
    labels.append(label)

num_samples = len(labels)
samples_lstm_periodic = np.reshape(np.array(samples_lstm_periodic),(num_samples,pre,1))
labels = np.array(labels)
print(samples_lstm_periodic.shape)
print(labels.shape)

#################### train and test split
x_train_lstm_periodic = samples_lstm_periodic[:4000]
x_test_lstm_periodic = samples_lstm_periodic[4000:]
y_train = labels[:4000]
y_test = labels[4000:]

print(x_train_lstm_periodic.shape,y_train.shape)
print(x_test_lstm_periodic.shape,y_test.shape)
plt.figure(figsize=(20,10))
plt.plot(y_test,'r')

################################################ Model: daily Block - Multi-STGCnet
input110 = Input(shape=(pre,1), dtype='float')
input111 = layers.LSTM(35,return_sequences=True)(input110)
input112 = layers.LSTM(pre)(input111)
output11 = layers.Dense(1,activation='relu')(input112)
model = models.Model(inputs=[input110],outputs=[output11])
model.summary()
model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse','mape'])
callbacks_list = [
    keras.callbacks.EarlyStopping(
    monitor='mae',
    patience=10,),
    keras.callbacks.ModelCheckpoint(filepath='lstm_periodic.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)

################################################ Model training
epochs = 1000
H = model.fit([x_train_lstm_periodic], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=([x_test_lstm_periodic],y_test))

################################################ Loss 
train_loss = H.history['loss']
test_loss = H.history['val_loss']
iterations = [i for i in range(epochs)]
plt.plot(iterations, train_loss,'b-',label='train_mae')
plt.plot(iterations, test_loss,'r-',label='test_mae')
plt.legend()
plt.title('Train_mae VS Test_mae')

################################################ predict
path="./lstm_periodic.h5"
model_best = models.load_model(path)
model.summary()
predict_best = model_best.predict(x_test_lstm_periodic)
