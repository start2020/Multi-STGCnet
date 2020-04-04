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
samples_lstm_recent = np.reshape(np.array(samples_lstm_recent),(num_samples,pre,1))
labels = np.array(labels)
print(samples_lstm_recent.shape)
print(labels.shape)


#################### train and test split
x_train_lstm_recent = samples_lstm_recent[:4000]
x_test_lstm_recent = samples_lstm_recent[4000:]
y_train = labels[:4000]
y_test = labels[4000:]

print(x_train_lstm_recent.shape,y_train.shape)
print(x_test_lstm_recent.shape,y_test.shape)

################################################ Model:GradientBoost
input110 = Input(shape=(pre,1), dtype='float')
input111 = layers.GRU(35,return_sequences=True)(input110)
input112 = layers.GRU(pre)(input111)
output11 = layers.Dense(1,activation='relu')(input112)
model = models.Model(inputs=[input110],outputs=[output11])
model.summary()
model.compile(optimizer='rmsprop',loss='mae',metrics=['mae','mse','mape'])
callbacks_list = [
    keras.callbacks.EarlyStopping(
    monitor='mae',
    patience=10,),
    keras.callbacks.ModelCheckpoint(filepath='GRU.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)
                 ]

################################################ Training
epochs = 1000
H = model.fit([x_train_lstm_recent], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=([x_test_lstm_recent],y_test))

################################################ Loss
train_loss = H.history['loss']
test_loss = H.history['val_loss']
iterations = [i for i in range(epochs)]
plt.plot(iterations, train_loss,'b-',label='train_mae')
plt.plot(iterations, test_loss,'r-',label='test_mae')
plt.legend()
plt.title('Train_mae VS Test_mae')

################################################ Predict
path="./GRU.h5"
model_best = models.load_model(path)
model.summary()
predict_best = model_best.predict(x_test_lstm_recent)
