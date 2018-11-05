
# coding: utf-8
# In[]:Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,confusion_matrix
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
# In[] Data_Aggregate_clip_Scal
    
#GlobalClipper = [] #先創建一個空list
#GlobalScaler = MinMaxScaler()
def Data_Aggregate_Clip_Scal(dataset_name_list, Clipper, Scaler):
    """放進模型前的資料處理:加總->clipping->scaling
    Argn:
    >dataset_name_list:要加總的各期資料的名稱字串
    >Clipper: 若有給定Clipper，則使用給定的Clipper來clipping；若Clipper== None，則重新fit一個Clipper
    >Scaler:若有給定Scaler，則使用給定的Scaler來scaling；若Scaler== None，則重新fit一個Scaler
    
    Return:
    >data_temp: 處理完的稀疏資料
    >Clipper: Clipper list
    >Scaler: Scaler 物件
    """
    #Aggregating
    data_temp = Item[ dataset_name_list[0] ]  #先取出list中的第一個時期
    for i in range( 1,len(dataset_name_list) ) : 
        data_temp = data_temp + Item[ dataset_name_list[i] ]  #加總剩下的每個時期
    #Clipping
    if Clipper == None :     #若未給定Clipper，則重新fit一個Clipper
        Clipper = [] #先創建一個空list
        for i in range(data_temp.shape[1]):
            nonzero = data_temp[ data_temp[:,i]>0, i] #篩出每一欄非0的值
            clip_temp = 0
            if len(nonzero)>0 : #每一欄至少要有一個非0的值,才clip，因為會出現某一總商品皆未被購買的情況
                clip_temp = np.percentile(nonzero, 75, axis=0) + (1.5*(np.percentile(nonzero, 75, axis=0) - np.percentile(nonzero, 25, axis=0)))
            Clipper.append( clip_temp  ) #Clip在Q3+1.5個IQR        
    data_temp = np.clip(data_temp, 0, Clipper)
    
    #Scaling
    if Scaler == None :   #若未給定Scaler，則重新fit一個Scaler
        Scaler = MinMaxScaler() #GlobalScaler:繼承MinMaxScaler，之後用以反轉(inverse_transform)
        Scaler.fit(data_temp)
    data_temp = Scaler.transform(data_temp)
    return  data_temp, Clipper, Scaler  

# In[] M1 Input Data current Q : 
#x = Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
 #   Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"]

train_x1, Clipper_train_x1, Scaler_train_x1 = Data_Aggregate_Clip_Scal(["M7WD","M8WD","M9WD","M7WE","M8WE","M9WE"], 
                                                     Clipper=None, Scaler=None)

train_y1, Clipper_train_y1, Scaler_train_y1 = Data_Aggregate_Clip_Scal(["M10WD","M10WE"], 
                                                     Clipper=None, Scaler=None)

valid_x1, _, _ = Data_Aggregate_Clip_Scal(["M10WD","M8WD","M9WD","M10WE","M8WE","M9WE"], 
                                                     Clipper=Clipper_train_x1, Scaler=Scaler_train_x1) # _代表不需要用到

valid_y1, _, Scaler_valid_y1 = Data_Aggregate_Clip_Scal(["M11WD","M11WE"], 
                                                     Clipper=Clipper_train_y1, Scaler=Scaler_train_y1)
# In[]:M1 relu +　mean_squared_error
A1 = Input(shape = (2404,),name='Input')
B = Dense(2000,activation='relu',)(A1)
B = Dropout(0.3)(B)
C = Dense(1000,activation='relu',)(B)
C = Dropout(0.3)(C)
N = Dense(300,activation='relu',)(C)
N = Dropout(0.3)(N)
O = Dense(1000,activation='relu',)(N)
O = Dropout(0.3)(O)
P = Dense(2000,activation='relu',)(O)
P = Dropout(0.3)(P)
y = Dense(2404, activation='relu',name='Output')(P)

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
# In[] M1 Input Data current Q : 
#x = Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
 #   Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"]

train_x1, Clipper_train_x1, Scaler_train_x1 = Data_Aggregate_Clip_Scal(["M7WD","M8WD","M9WD","M7WE","M8WE","M9WE"], 
                                                     Clipper=None, Scaler=None)

train_y1, Clipper_train_y1, Scaler_train_y1 = Data_Aggregate_Clip_Scal(["M10WD","M10WE"], 
                                                     Clipper=None, Scaler=None)

valid_x1, _, _ = Data_Aggregate_Clip_Scal(["M10WD","M8WD","M9WD","M10WE","M8WE","M9WE"], 
                                                     Clipper=Clipper_train_x1, Scaler=Scaler_train_x1) # _代表不需要用到

valid_y1, _, Scaler_valid_y1 = Data_Aggregate_Clip_Scal(["M11WD","M11WE"], 
# In[]:M1 sigmoid + binary_crossentropy
A1 = Input(shape = (2404,),name='Input')
B = Dense(2000,activation='relu',)(A1)
B = Dropout(0.3)(B)
C = Dense(1000,activation='relu',)(B)
C = Dropout(0.3)(C)
N = Dense(300,activation='relu',)(C)
N = Dropout(0.3)(N)
O = Dense(1000,activation='relu',)(N)
O = Dropout(0.3)(O)
P = Dense(2000,activation='relu',)(O)
P = Dropout(0.3)(P)
y = Dense(2404, activation='relu',name='Output')(P)

model = Model(inputs=[A1], outputs=[y])
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
model.compile(loss = 'binary_crossentropy', optimizer =  adagrad,
              metrics=['mean_squared_error','mean_absolute_error'])
model.summary()
modelcallbacks = model.fit(x = train_x1, y = train_y1,
                epochs=5, batch_size=64, shuffle=True,
                validation_data=(valid_x1, valid_y1,))


# Test data performance
#testPerformance = 
model.evaluate(valid_x1, valid_y1,
               batch_size=None, verbose=1, sample_weight=None, steps=None)

# In[]: Summarize history for loss
plt.figure()
plt.plot(modelcallbacks.history['loss'])  #loss mean_absolute_error 
plt.plot(modelcallbacks.history['val_loss'])  #val_loss val_mean_absolute_error
plt.plot(modelcallbacks.history['mean_absolute_error'])  #loss mean_absolute_error 
plt.plot(modelcallbacks.history['val_mean_absolute_error'])  #val_loss val_mean_absolute_error
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error'], loc='upper left')
plt.show()

# In[]: 反轉 (inverse_transform)
train_y1_inverse_transform = Scaler_train_y1.inverse_transform( train_y1 )
train_y1_hat = Scaler_train_y1.inverse_transform( model.predict(train_x1) )

valid_y1_inverse_transform = Scaler_valid_y1.inverse_transform( valid_y1 )
valid_y1_hat = Scaler_valid_y1.inverse_transform( model.predict(valid_y1) )

# In[]: M1_performance_table
df2 = pd.DataFrame(np.random.randint(low=0, high=2, size=(5, 3))).values
# 計算法一 
"""temp = []
for i in range((train_y1.shape)[0]):
    mb_count = 0
    for j in range((train_y1_hat.shape)[1]):
        if (train_y1[i,j] > 0) :
            mb_count = mb_count+1
    temp.append(mb_count)

temp2=[ ]
for i in range((train_y1_hat.shape)[0]):
    mb_count = 0
    for j in range((train_y1_hat.shape)[1]):
        if (train_y1[i,j] > 0) & (train_y1_hat[i,j] > 0) :
            mb_count = mb_count+1
    temp2.append(mb_count)"""
    
 # 計算法二 : 藉由confusion matrix 
cm_temp = [] #先創建一個LIST
for i in range(0,(train_y1.shape)[0]): #每一列(會員)，用實際的Item vs 預測的Item，來計算一個confusion matrix
    cm = confusion_matrix(y_true = np.where(train_y1[i,:] > 0, 1, 0),
                     y_pred = np.where(train_y1_hat[i,:] > 0, 1, 0))
    if cm.shape == (2, 2): #防止某些會員Item購買量全為0，若發生此狀況，則該會員的performance不記錄
        cm_temp.append([(cm[1,0] +  cm[1,1]) , cm[1,1]])
    if i % 1000 == 0 :
        print(i)
    
M1_performance_table = pd.DataFrame(data=cm_temp, columns={"#True1","Predict1_of_true1"})

# In[]:Scatterplot
#sns.scatterplot(x="#True1", y="Predict1_of_true1", data=M1_performance_table, marker =  ".") 
plt.scatter(x="#True1", y="Predict1_of_true1", data=M1_performance_table, marker =  ".",
            color='blue', s=1) 
plt.ylim(0, 100)
plt.xlim(0, 100)
#對角線
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()
    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
p1 = [0,0]
p2 = [500,500]
newline(p1,p2)

#趨勢線
z = np.polyfit(M1_performance_table["#True1"] , M1_performance_table["Predict1_of_true1"], 1)
p = np.poly1d(z)
plt.plot(M1_performance_table["#True1"],p(M1_performance_table["#True1"]),"r--")
print("y=%.6fx+(%.6f)"%(z[0],z[1])) #趨勢線方程式
plt.savefig('filename.png', dpi = 300)
plt.show()
# In[]:Scatterplot
#sns.scatterplot(x="#True1", y="Predict1_of_true1", data=M1_performance_table, marker =  ".") 
plt.scatter(x="#True1", y="Predict1_of_true1", data=M1_performance_table, marker =  ".",
            color='blue', s=1) 
plt.ylim(0, 500)
plt.xlim(0, 500)
#對角線
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()
    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l
p1 = [0,0]
p2 = [500,500]
newline(p1,p2)

#趨勢線
z = np.polyfit(M1_performance_table["#True1"] , M1_performance_table["Predict1_of_true1"], 1)
p = np.poly1d(z)
plt.plot(M1_performance_table["#True1"],p(M1_performance_table["#True1"]),"r--")
print("y=%.6fx+(%.6f)"%(z[0],z[1])) #趨勢線方程式
plt.savefig('filename.png', dpi = 300)
plt.show()


# In[]:plot_model
from keras.utils.vis_utils import plot_model 
import os
os.environ["path"] += os.pathsep +'C:/Users/e3621/AppData/Local/conda/conda/envs/tensorflow_yuta/Library/bin/graphviz/'
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model, show_shapes =True ).create(prog='dot', format='svg'))

# In[]
# In[]
# In[]
# In[]
# In[]
# In[]
# In[]
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
