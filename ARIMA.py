import numpy as np
import pandas as pd
from keras import models, layers, Input
import matplotlib.pyplot as plt
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import  stats
from statsmodels.graphics.api import qqplot
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

plt.figure(figsize=(20,10))
plt.plot(y_test,'r')

################################################ Model
y_test = pd.Series(y_test)
d = 1
diff = y_test.diff(d).dropna()
print(y_test.shape)
print(diff.shape)
order =(4,2)
#ARMA
ARMAmodel = sm.tsa.ARMA(diff, order).fit()

################################################ Model training
epochs = 1000
H = model.fit([x_train_lstm_daily], y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=([x_test_lstm_daily],y_test))

################################################ Loss 


################################################ predict
predicts = y_test.values[:-d] + ARMAmodel.fittedvalues
timeseries = y_test[d:].values
print(predicts.shape)
print(timeseries.shape)

path = "./results.xlsx"
df = pd.read_excel(path)
df['arima'] = predicts*9900
df.to_excel(path,index = False)

plt.figure(figsize=(20,10))
plt.plot(timeseries,'r')
plt.plot(predicts,'g')
