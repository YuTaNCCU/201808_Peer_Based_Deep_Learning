
# coding: utf-8
# In[]:Import packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from keras import metrics, optimizers, regularizers, backend as K
from keras.layers import Input, Dense, concatenate, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.utils import plot_model
K.clear_session()
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
# In[]: Load Data
#MbItemSparseTable = {} 
Item = {} #縮短變數名稱
for month in range(1,13):
   for weekday in ["WD","WE"]:
       str_temp = "M" + str(month) + str(weekday) #創建DICTIONARY 的KEY    
       Item[str_temp] = pd.read_csv("data/"+str_temp + ".csv").drop(columns=["Unnamed: 0","MbID"]).values 
     
#暫時處理 #{'1000095', '1000154', '1000285', '1000913'}

# In[]:紀錄Performance的DataFrame
M1_MSE_DF = pd.DataFrame(data={"Null":[0,0,0], "RFM":[0,0,0],  "RFMCB":[0,0,0], 
                               "RFMCBGW":[0,0,0],  "RFMCBGWE":[0,0,0]}, index =["Train","Valid","Test"])
M1_MAE_DF = pd.DataFrame(data={"Null":[0,0,0], "RFM":[0,0,0],  "RFMCB":[0,0,0], 
                               "RFMCBGW":[0,0,0],  "RFMCBGWE":[0,0,0]}, index =["Train","Valid","Test"])
    
# In[] Input Data current Q : 
#x = Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
 #   Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"]

min_max_scaler = MinMaxScaler()

train_x1 = Item["M7WD"]+ Item["M8WD"]+ Item["M9WD"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]
train_x1 = MinMaxScaler().fit_transform(train_x1)

valid_x1 = Item["M10WD"]+ Item["M8WD"]+ Item["M9WD"]+ Item["M10WE"]+ Item["M8WE"]+ Item["M9WE"] 
valid_x1 = MinMaxScaler().fit_transform(valid_x1)

train_y1 = Item["M10WD"]+ Item["M10WE"] 
scaler_train_y1 = MinMaxScaler().fit(train_y1) #繼承MinMaxScaler，之後用以反轉(inverse_transform)
train_y1 = scaler_train_y1.transform(train_y1)

valid_y1 = Item["M11WD"]+ Item["M11WE"] 
scaler_valid_y1 = MinMaxScaler().fit(valid_y1)
valid_y1 = scaler_valid_y1.transform(valid_y1)


# In[]:M1
A1 = Input(shape = (2404,))
B = Dense(300,activation='relu',)(A1)
B = Dropout(0.2)(B)
N = Dense(300,activation='relu',)(B)
N = Dropout(0.2)(N)
y = Dense(2404, activation='sigmoid',name='Dense4')(N)

model = Model(inputs=[A1], outputs=[y])
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss = 'mean_squared_error', optimizer =  adagrad,
              metrics=['mean_absolute_error'])
model.summary()
modelcallbacks = model.fit(x = train_x1, y = train_y1,
                epochs=5, batch_size=64, shuffle=True,
                validation_data=(valid_x1, valid_y1,))


# Test data performance
#testPerformance = 
model.evaluate(valid_x1, valid_y1,
               batch_size=None, verbose=1, sample_weight=None, steps=None)
# In[]: 反轉(inverse_transform)

train_y1_hat = scaler_train_y1.inverse_transform( model.predict(train_x1) )

valid_y1_hat = scaler_valid_y1.inverse_transform( model.predict(valid_y1) )

# In[]:  summarize history for loss
plt.figure()
plt.plot(modelcallbacks.history['mean_absolute_error'])  #loss mean_absolute_error 
plt.plot(modelcallbacks.history['val_mean_absolute_error'])  #val_loss val_mean_absolute_error
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

# In[]:plot_model
from keras.utils.vis_utils import plot_model 
import os
os.environ["path"] += os.pathsep +'C:/Users/e3621/AppData/Local/conda/conda/envs/tensorflow_yuta/Library/bin/graphviz/'
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model, show_shapes =True ).create(prog='dot', format='svg'))

# In[] Input Data current Q + Cumulation : 
train_x2 = ( Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
            Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"]+ \
            Item["M1WD"]+ Item["M2WD"]+ Item["M3WD"]+ Item["M4WD"]+ Item["M5WD"]+ \
            Item["M6WD"]+ Item["M7WD"]+ Item["M8WD"]+ Item["M9WD"]+ Item["M10WD"] \
            ).values
train_x2 = min_max_scaler.fit_transform(train_x2)
    
valid_x2 = ( Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
            Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"]+ Item["M11WE"]+ \
            Item["M1WD"]+ Item["M2WD"]+ Item["M3WD"]+ Item["M4WD"]+ Item["M5WD"]+ \
            Item["M6WD"]+ Item["M7WD"]+ Item["M8WD"]+ Item["M9WD"]+ Item["M10WD"]+ Item["M11WD"]\
            ).values
valid_x2 = min_max_scaler.fit_transform(valid_x2)
# In[]:M2
A1 = Input(shape = (2404,))
A2 = Input(shape = (2404,))
B1 = Dense(300,activation='relu')(A1)
B2 = Dense(300,activation='relu')(A2)
B1 = Dropout(0.2)(B1)
B2 = Dropout(0.2)(B2)
Merged = concatenate([B1, B2])
N = Dense(300,activation='relu',)(Merged)
N = Dropout(0.2)(N)
y = Dense(2404, activation='softmax',name='Dense4')(N)

model = Model(inputs=[A1,A2], outputs=[y])
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss = 'mean_squared_error', optimizer =  adagrad,
              metrics=['mean_absolute_error'])
model.summary()
modelcallbacks = model.fit(x = [train_x1,train_x2], y = train_y1,
                epochs=15, batch_size=64, shuffle=True,
                validation_data=([valid_x1,valid_x2], valid_y1,))


# Test data performance
#testPerformance = 
"""model.evaluate([valid_x1,valid_x2], valid_y1,
               batch_size=None, verbose=1, sample_weight=None, steps=None)"""


# In[] current Q + Cumulation + 工作日\假日 :
    
train_x1_WE = (Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]).values
train_x1_WD = (Item["M7WD"]+ Item["M8WD"]+ Item["M9WD"]).values
train_x2_WE = ( Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
            Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"] ).values
train_x2_WD = ( Item["M1WD"]+ Item["M2WD"]+ Item["M3WD"]+ Item["M4WD"]+ Item["M5WD"]+ \
            Item["M6WD"]+ Item["M7WD"]+ Item["M8WD"]+ Item["M9WD"]+ Item["M10WD"] ).values

valid_x1_WE = (Item["M10WE"]+ Item["M8WE"]+ Item["M9WE"]).values
valid_x1_WD = (Item["M10WD"]+ Item["M8WD"]+ Item["M9WD"]).values                  
valid_x2_WE = ( Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
            Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"]+ Item["M11WE"] ).values
valid_x2_WD = ( Item["M1WD"]+ Item["M2WD"]+ Item["M3WD"]+ Item["M4WD"]+ Item["M5WD"]+ \
            Item["M6WD"]+ Item["M7WD"]+ Item["M8WD"]+ Item["M9WD"]+ Item["M10WD"]+ Item["M11WD"] ).values               
# In[]:M3
A1_WE = Input(shape = (2404,))
A1_WD = Input(shape = (2404,))
A2_WE = Input(shape = (2404,))
A2_WD = Input(shape = (2404,))
B1_WE = Dense(300,activation='relu')(A1_WE)
B1_WD = Dense(300,activation='relu')(A1_WD)
B2_WE = Dense(300,activation='relu')(A2_WE)
B2_WD = Dense(300,activation='relu')(A2_WD)
B1_WE = Dropout(0.2)(B1_WE)
B1_WD = Dropout(0.2)(B1_WD)
B2_WE = Dropout(0.2)(B2_WE)
B2_WD = Dropout(0.2)(B2_WD)
Merged = concatenate([B1_WE, B1_WD, B2_WE , B2_WD])
N = Dense(300,activation='relu',)(Merged)
N = Dropout(0.2)(N)
y = Dense(2404, activation='relu',name='Dense4')(N)

model = Model(inputs=[A1_WE, A1_WD, A2_WE, A2_WD], outputs=[y])
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss = 'mean_squared_error', optimizer =  adagrad,
              metrics=['mean_absolute_error'])
model.summary()
modelcallbacks = model.fit(x = [train_x1_WE,train_x1_WD,train_x2_WE,train_x2_WD], y = train_y1,
                epochs=15, batch_size=64, shuffle=True,
                validation_data=([valid_x1_WE,valid_x1_WD,valid_x2_WE,valid_x2_WD], valid_y1,))


# Test data performance
#testPerformance = 
model.evaluate([train_x1_WE,train_x1_WD,train_x2_WE,train_x2_WD], valid_y1,
               batch_size=None, verbose=1, sample_weight=None, steps=None)

# In[]:M3 Softmax
A1_WE = Input(shape = (2404,))
A1_WD = Input(shape = (2404,))
A2_WE = Input(shape = (2404,))
A2_WD = Input(shape = (2404,))
B1_WE = Dense(300,activation='relu')(A1_WE)
B1_WD = Dense(300,activation='relu')(A1_WD)
B2_WE = Dense(300,activation='relu')(A2_WE)
B2_WD = Dense(300,activation='relu')(A2_WD)
B1_WE = Dropout(0.2)(B1_WE)
B1_WD = Dropout(0.2)(B1_WD)
B2_WE = Dropout(0.2)(B2_WE)
B2_WD = Dropout(0.2)(B2_WD)
Merged = concatenate([B1_WE, B1_WD, B2_WE , B2_WD])
N = Dense(300,activation='relu',)(Merged)
N = Dropout(0.2)(N)
y = Dense(2404, activation='softmax',name='Dense4')(N)

model = Model(inputs=[A1_WE, A1_WD, A2_WE, A2_WD], outputs=[y])
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss = 'mean_squared_error', optimizer =  adagrad,
              metrics=['mean_absolute_error'])
model.summary()
modelcallbacks = model.fit(x = [train_x1_WE,train_x1_WD,train_x2_WE,train_x2_WD], y = train_y1,
                epochs=15, batch_size=64, shuffle=True,
                validation_data=([valid_x1_WE,valid_x1_WD,valid_x2_WE,valid_x2_WD], valid_y1,))


# Test data performance
#testPerformance = 
"""model.evaluate([train_x1_WE,train_x1_WD,train_x2_WE,train_x2_WD], valid_y1,
               batch_size=None, verbose=1, sample_weight=None, steps=None)"""


# In[]: 
# In[]: 
# In[]: Test

M2_MSE_DF.loc["Train","RFMCBGWE"]=modelcallbacks.history['loss'][-1]
M2_MSE_DF.loc["Valid","RFMCBGWE"]=modelcallbacks.history['val_loss'][-1]
M2_MSE_DF.loc["Test","RFMCBGWE"]=testPerformance[0]

M2_MAE_DF.loc["Train","RFMCBGWE"]=modelcallbacks.history['mean_absolute_error'][-1]
M2_MAE_DF.loc["Valid","RFMCBGWE"]=modelcallbacks.history['val_mean_absolute_error'][-1]
M2_MAE_DF.loc["Test","RFMCBGWE"]=testPerformance[1]
display(M2_MSE_DF,M2_MAE_DF)

## MSE MAE barplot
plt.figure()
stacked = M3_MSE_DF.stack().reset_index().rename(columns={0:'MSE',"level_0": "Set", "level_1": "Features",})
sns.barplot(x=stacked.Features, y=stacked.MSE, hue=stacked.Set)

plt.figure()
stacked = M3_MAE_DF.stack().reset_index().rename(columns={0:'MAE',"level_0": "Set", "level_1": "Features",})
sns.barplot(x=stacked.Features, y=stacked.MAE, hue=stacked.Set)

M = "RFMCBGWE" #模型的名稱，對應DataFrame
### Train
y_predict = model.predict(x = [Q1_X.loc[:,["R","F","M","Breadth", 'L2', 'Entropy', 'LogUtility', 'C3', 'age', 'gender_0F1M',
                           'CountSun', 'CountMon', 'CountTue', 'CountWed', 'CountThu', 'CountFri', 'CountSat' ]].values,
               Q1_Item_2000.iloc[:,1:].values])
y_predict01 =np.where(y_predict >=0.58, 1, 0)
y_true = Q2_Y.loc[:,"F_CLipped"].values
y_true01 =np.where( y_true> 0, 1, 0)
M3_Acc_DF.loc["Train",M] = accuracy_score(y_true= y_true01, y_pred =  y_predict01, normalize=True)
#### densityplt
plt.figure()
sns.distplot(y_true.reshape(75393)- y_predict.round().reshape(75393), color = 'b', hist=False)

### Valid
y_predict = model.predict(x = [Q2_X.loc[:,["R","F","M","Breadth", 'L2', 'Entropy', 'LogUtility', 'C3', 'age', 'gender_0F1M',
                           'CountSun', 'CountMon', 'CountTue', 'CountWed', 'CountThu', 'CountFri', 'CountSat' ]].values,
               Q2_Item_2000.iloc[:,1:].values])
y_predict01 =np.where(y_predict >=0.58, 1, 0)
y_true = Q3_Y.loc[:,"F_CLipped"].values
y_true01 =np.where( y_true> 0, 1, 0)
M3_Acc_DF.loc["Valid",M] = accuracy_score(y_true= y_true01, y_pred =  y_predict01, normalize=True)
#### densityplt
sns.distplot(y_true.reshape(75393)- y_predict.round().reshape(75393), color = 'orange', hist=False)

### Test
y_predict = model.predict(x = [Q3_X.loc[:,["R","F","M","Breadth", 'L2', 'Entropy', 'LogUtility', 'C3', 'age', 'gender_0F1M',
                           'CountSun', 'CountMon', 'CountTue', 'CountWed', 'CountThu', 'CountFri', 'CountSat' ]].values,
               Q3_Item_2000.iloc[:,1:].values])
y_predict01 =np.where(y_predict >=0.58, 1, 0)
y_true = Q4_Y.loc[:,"F_CLipped"].values
y_true01 =np.where( y_true> 0, 1, 0)
M3_Acc_DF.loc["Test",M] = accuracy_score(y_true= y_true01, y_pred =  y_predict01, normalize=True)
#### densityplt
sns.distplot(y_true.reshape(75393)- y_predict.round().reshape(75393), color = 'g', hist=False)

#### Acc barplot
display(M3_Acc_DF)
plt.figure()
stacked = M3_Acc_DF.stack().reset_index().rename(columns={0:'Accuracy',"level_0": "Set", "level_1": "Features",})
sns.barplot(x=stacked.Features, y=stacked.Accuracy, hue=stacked.Set)
