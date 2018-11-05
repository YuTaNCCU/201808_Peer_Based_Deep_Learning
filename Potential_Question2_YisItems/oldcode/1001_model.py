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
from keras import metrics, optimizers, regularizers, losses,backend as K
from keras.layers import Input, Dense, concatenate, Dropout, LeakyReLU
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import datetime
# In[]: Load Data
Item1 = {} #品號
Item2 = {} #群號
Item3 = {} #單品
for month in range(1,13):
   for weekday in ["WD","WE"]:
       str_temp = "M" + str(month) + str(weekday) #創建DICTIONARY 的KEY    
       Item1[str_temp] = pd.read_csv("data/"+'FirstNumber'+str_temp + ".csv").drop(columns=["Unnamed: 0","MbID"]).values  #讀取品號First Number
       Item2[str_temp] = pd.read_csv("data/"+'SecondNumber'+str_temp + ".csv").drop(columns=["Unnamed: 0","MbID"]).values  #讀取群號Second Number
       Item3[str_temp] = pd.read_csv("data/"+str_temp + ".csv").drop(columns=["Unnamed: 0","MbID"]).values  #讀取單品
# In[]:紀錄Performance的DataFrame
M1_MSE_DF = pd.DataFrame(data={"Null":[0,0,0], "RFM":[0,0,0],  "RFMCB":[0,0,0], 
                               "RFMCBGW":[0,0,0],  "RFMCBGWE":[0,0,0]}, index =["Train","Valid","Test"])
M1_MAE_DF = pd.DataFrame(data={"Null":[0,0,0], "RFM":[0,0,0],  "RFMCB":[0,0,0], 
                               "RFMCBGW":[0,0,0],  "RFMCBGWE":[0,0,0]}, index =["Train","Valid","Test"])
# In[] Data_Aggregate_clip_Scal
#GlobalClipper = [] #先創建一個空list
#GlobalScaler = MinMaxScaler()
def Data_Aggregate_Clip_Scal(dataset_name_list,  Dataset, Clipper, Scaler):
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
    data_temp = Dataset[ dataset_name_list[0] ]  #先取出list中的第一個時期
    for i in range( 1,len(dataset_name_list) ) : 
        data_temp = data_temp + Dataset[ dataset_name_list[i] ]  #加總剩下的每個時期
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

# In[]: Scatterplot Function
#圖1 Recall：  橫軸:實際有買的商品數；縱軸:實際有買的商品中預測有買的商品數
#圖2 Precision：  橫軸:預測有買的商品數；縱軸:預測有買的商品中實際有買的商品數
def Scatterplot_RP( xTrue, yPredict, xTrue_NUll, yPredict_NUll, lower = 0, upper = 500) : 
    """xTrue = a
    yPredict = b
    xTrue_NUll = an
    yPredict_NUll = bn
    limit = 500"""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    #sns.scatterplot(x="#True1", y="Predict1_of_true1", data=M1_performance_table, marker =  ".") 
    plt.scatter(x=xTrue_NUll, y=yPredict_NUll, marker =  ".", color='green', s=1) 
    plt.scatter(x=xTrue, y=yPredict, marker =  ".", color='blue', s=1) 
    plt.xticks(np.arange(lower, upper+1, (upper-lower)/5))#軸刻度的間隔
    plt.yticks(np.arange(lower, upper+1, (upper-lower)/10))
    plt.ylim(lower, upper)#軸的極值
    plt.xlim(lower, upper)
    #畫一條對角線
    def newline(p1, p2):
        ax = plt.gca()
        xmin, xmax = ax.get_xbound()
        if(p2[0] == p1[0]):
            xmin = xmax = p1[0]
            ymin, ymax = ax.get_ybound()
        else:
            ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
            ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
        l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='black')
        ax.add_line(l)
        return l
    p1 = [0,0]
    p2 = [500,500]
    newline(p1,p2)
    #趨勢線 Null
    z = np.polyfit(xTrue_NUll , yPredict_NUll, 1)
    p = np.poly1d(z)
    plt.plot(xTrue,p(xTrue),"r--", color='green')
    print("Null Model: y=%.6fx+(%.6f)"%(z[0],z[1])) #趨勢線方程式
    #趨勢線
    z = np.polyfit(xTrue , yPredict, 1)
    p = np.poly1d(z)
    plt.plot(xTrue,p(xTrue),"r--", color='blue')
    print("My Model: y=%.6fx+(%.6f)"%(z[0],z[1])) #趨勢線方程式
    #網格線
    plt.grid()
    plt.savefig( 'Scatterplot_RP' + str(lower) + '_' + str(upper) + '.png', dpi = 300)
    plt.show()
# In[]: Scatterplot Function
#Number_not_in_NullModel v.s. Number_not_in_NullModel_but_predict     
def Scatterplot_CompareWithNull( xTrue, yPredict, xlabel, ylabel, lower = 0, upper = 500) : 
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    plt.scatter(x=xTrue, y=yPredict, marker =  ".", color='red', s=1) 
    plt.xticks(np.arange(lower, upper+1, (upper-lower)/5))#軸刻度的間隔
    plt.yticks(np.arange(lower, upper+1, (upper-lower)/10))
    plt.ylim(lower, upper)#軸的極值
    plt.xlim(lower, upper)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #畫一條對角線
    def newline(p1, p2):
        ax = plt.gca()
        xmin, xmax = ax.get_xbound()
        if(p2[0] == p1[0]):
            xmin = xmax = p1[0]
            ymin, ymax = ax.get_ybound()
        else:
            ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
            ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
        l = mlines.Line2D([xmin,xmax], [ymin,ymax], color='black')
        ax.add_line(l)
        return l
    p1 = [0,0]
    p2 = [500,500]
    newline(p1,p2)
    #趨勢線
    z = np.polyfit(xTrue , yPredict, 1)
    p = np.poly1d(z)
    plt.plot(xTrue,p(xTrue),"r--", color='red')
    print("My Model: y=%.6fx+(%.6f)"%(z[0],z[1])) #趨勢線方程式
    #網格線
    plt.grid()
    plt.savefig( 'Scatterplot_CompareWithNull' + str(lower) + '_' + str(upper) + '.png', dpi = 300)
    plt.show()
    
Scatterplot_CompareWithNull( xTrue = a, yPredict = b,xlabel=" #Item not in Null Model ", ylabel = " #(Item not in Null Model but predicted) ", lower = 0, upper = 50 ) 
# In[] M1 data : Input Data current Q : 
#x = Item["M1WE"]+ Item["M2WE"]+ Item["M3WE"]+ Item["M4WE"]+ Item["M5WE"]+ \
 #   Item["M6WE"]+ Item["M7WE"]+ Item["M8WE"]+ Item["M9WE"]+ Item["M10WE"]

train_x3, Clipper_train_x3, Scaler_train_x3 = Data_Aggregate_Clip_Scal(["M7WD","M8WD","M9WD","M7WE","M8WE","M9WE"], 
                                                     Dataset = Item3, Clipper=None, Scaler=None)

train_y3, Clipper_train_y3, Scaler_train_y3 = Data_Aggregate_Clip_Scal(["M10WD","M10WE"], 
                                                     Dataset = Item3 ,Clipper=None, Scaler=None)

#np.savetxt("train_y3.csv", train_y3, delimiter=",") #輸出CSV

valid_x3, _, _ = Data_Aggregate_Clip_Scal(["M10WD","M8WD","M9WD","M10WE","M8WE","M9WE"], 
                                                     Dataset = Item3, Clipper=Clipper_train_x3, Scaler=Scaler_train_x3) # _代表不需要用到

valid_y3, _, Scaler_valid_y3 = Data_Aggregate_Clip_Scal(["M11WD","M11WE"], 
                                                     Dataset = Item3, Clipper=Clipper_train_y3, Scaler=Scaler_train_y3)
#np.savetxt("valid_y3.csv", valid_y3, delimiter=",") #輸出CSV
# In[]:M1 
A1 = Input(shape = (2404,),name='Input')
B = Dense(1000,activation='relu')(A1)
B = Dropout(0.4)(B)
C = Dense(500,activation='relu',)(B)
C = Dropout(0.4)(C)
N = Dense(200,activation='relu',)(C)
N = Dropout(0.4)(N)
O = Dense(500,activation='relu',)(N)
O = Dropout(0.4)(O)
P = Dense(1000,activation='relu',)(O)
P = Dropout(0.4)(P)
y = Dense(2404, activation='relu',name='Output')(P)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
adagrad = optimizers.Adagrad(lr=0.0002, epsilon=None, decay=0.0)
def my_mean_squared_error(y_true, y_pred) : #放大mean_squared_error
    myloss = losses.mean_squared_error(y_true, y_pred) *500
    return myloss
def my_mean_absolute_error(y_true, y_pred) : #放大mean_squared_error
    myloss = losses.mean_absolute_error(y_true, y_pred) *500
    return myloss
model = Model(inputs=[A1], outputs=[y])
model.compile(loss = my_mean_squared_error, optimizer =  adam, metrics=[my_mean_absolute_error]) #mean_absolute_error
model.summary()
modelcallbacks = model.fit(x = train_x3, y = train_y3,
                epochs=5, batch_size=64, shuffle=True,
                validation_data=(valid_x3, valid_y3,))
#Save & Load weights
model.save(  'M1_model_weights_'+ str(datetime.datetime.now() ).replace(" ", "-").replace(":", "_").replace(".", "_")+'.h5'  )
#model=load_model('save_M1_model_weights_2018-09-20-08_57_45_810670.h5',custom_objects={'my_mean_squared_error': my_mean_squared_error, 'my_mean_absolute_error':my_mean_absolute_error})

# In[] M2 data : Input Data current Q + Cumulation :
train_x1, Clipper_train_x1, Scaler_train_x1 = Data_Aggregate_Clip_Scal(["M7WD","M8WD","M9WD","M7WE","M8WE","M9WE"], 
                                                     Dataset = Item1,  Clipper=None, Scaler=None)
valid_x1, _, _ = Data_Aggregate_Clip_Scal(["M10WD","M8WD","M9WD","M10WE","M8WE","M9WE"], 
                                                     Dataset = Item1,  Clipper=Clipper_train_x1, Scaler=Scaler_train_x1) 
train_x2, Clipper_train_x2, Scaler_train_x2 = Data_Aggregate_Clip_Scal(["M10WD","M8WD","M9WD","M10WE","M8WE","M9WE"], 
                                                     Dataset = Item2,  Clipper=None, Scaler=None)
valid_x2, _, _ = Data_Aggregate_Clip_Scal(["M10WD","M8WD","M9WD","M10WE","M8WE","M9WE"], 
                                                     Dataset = Item2,  Clipper=Clipper_train_x2, Scaler=Scaler_train_x2) 
# In[]:M2
A1 = Input(shape = (99,),name='Input_1')
A2 = Input(shape = (424,),name='Input_2')
A3 = Input(shape = (2404,),name='Input_3')
Merged = concatenate([A1, A2, A3])
B = Dense(1000,activation='relu')(Merged)
B = Dropout(0.4)(B)
C = Dense(500,activation='relu',)(B)
C = Dropout(0.4)(C)
N = Dense(200,activation='relu',)(C)
N = Dropout(0.4)(N)
O = Dense(500,activation='relu',)(N)
O = Dropout(0.4)(O)
P = Dense(1000,activation='relu',)(O)
P = Dropout(0.4)(P)
y = Dense(2404, activation='relu',name='Output')(P)

model = Model(inputs=[A1,A2,A3], outputs=[y])
model.compile(loss = my_mean_squared_error, optimizer =  adam, metrics=[my_mean_absolute_error]) #mean_absolute_error
model.summary()
modelcallbacks = model.fit(x = [train_x1,train_x2,train_x3], y = train_y3,
                epochs=10, batch_size=64, shuffle=True,
                validation_data=([valid_x1,valid_x2,valid_x3], valid_y3,))

model.save(  'M2_model_weights_'+ str(datetime.datetime.now() ).replace(" ", "-").replace(":", "_").replace(".", "_")+'.h5'  )
model=load_model('save_M2_model_weights_2018-09-20-08_55_26_413114.h5',custom_objects={'my_mean_squared_error': my_mean_squared_error, 'my_mean_absolute_error':my_mean_absolute_error})

# In[] M3 data : 
list_temp = ["M1WE","M2WE","M3WE","M4WE","M5WE","M6WE","M7WE","M8WE","M9WE","M10WE", \
            "M1WD","M2WD","M3WD","M4WD","M5WD", "M6WD","M7WD","M8WD","M9WD","M10WD"] 

train_x1_cul, Clipper_train_x1_cul, Scaler_train_x1_cul = Data_Aggregate_Clip_Scal(dataset_name_list = list_temp, 
                                                     Dataset = Item1,  Clipper=None, Scaler=None)
train_x2_cul, Clipper_train_x2_cul, Scaler_train_x2_cul = Data_Aggregate_Clip_Scal(dataset_name_list = list_temp, 
                                                     Dataset = Item2,  Clipper=None, Scaler=None)
train_x3_cul, Clipper_train_x3_cul, Scaler_train_x3_cul = Data_Aggregate_Clip_Scal(dataset_name_list = list_temp, 
                                                     Dataset = Item3,  Clipper=None, Scaler=None)
list_temp = ["M1WE","M2WE","M3WE","M4WE","M5WE","M6WE","M7WE","M8WE","M9WE","M10WE","M11WE",\
            "M1WD","M2WD","M3WD","M4WD","M5WD", "M6WD","M7WD","M8WD","M9WD","M10WD","M11WD"] 
valid_x1_cul, _, _ = Data_Aggregate_Clip_Scal(dataset_name_list = list_temp, 
                                                     Dataset = Item1,  Clipper=Clipper_train_x1_cul, Scaler=Scaler_train_x1_cul) 
valid_x2_cul, _, _ = Data_Aggregate_Clip_Scal(dataset_name_list = list_temp, 
                                                     Dataset = Item2,  Clipper=Clipper_train_x2_cul, Scaler=Scaler_train_x2_cul) 
valid_x3_cul, _, _ = Data_Aggregate_Clip_Scal(dataset_name_list = list_temp, 
                                                     Dataset = Item3,  Clipper=Clipper_train_x3_cul, Scaler=Scaler_train_x3_cul) 

            
# In[]:M3
A1 = Input(shape = (99,),name='Input_1')
A2 = Input(shape = (424,),name='Input_2')
A3 = Input(shape = (2404,),name='Input_3')
A1_cul = Input(shape = (99,),name='Input_1_cul')
A2_cul = Input(shape = (424,),name='Input_2_cul')
A3_cul = Input(shape = (2404,),name='Input_3_cul')
Merged = concatenate([A1, A2, A3, A1_cul, A2_cul, A3_cul ])
B = Dense(2000,activation='LeakyReLU')(Merged)
B = Dropout(0.4)(B)
C = Dense(1000,activation='LeakyReLU',)(B)
C = Dropout(0.4)(C)
D = Dense(500,activation='elu',)(C)
D = Dropout(0.4)(D)
N = Dense(200,activation='relu',)(D)
N = Dropout(0.4)(N)
O = Dense(500,activation='relu',)(N)
O = Dropout(0.4)(O)
P = Dense(1000,activation='relu',)(O)
P = Dropout(0.4)(P)
y = Dense(2404, activation='relu',name='Output')(P)

model = Model(inputs=[A1, A2, A3, A1_cul, A2_cul, A3_cul ], outputs=[y])
model.compile(loss = my_mean_squared_error, optimizer =  adam, metrics=[my_mean_absolute_error]) #mean_absolute_error
model.summary()
modelcallbacks = model.fit(x = [train_x1,train_x2,train_x3,train_x1_cul,train_x2_cul,train_x3_cul], y = train_y3,
                epochs=5, batch_size=64, shuffle=True,
                validation_data=([valid_x1,valid_x2,valid_x3,valid_x1_cul,valid_x2_cul,valid_x3_cul], valid_y3,))

model.save(  'M2_model_weights_'+ str(datetime.datetime.now() ).replace(" ", "-").replace(":", "_").replace(".", "_")+'.h5'  )
#model=load_model('save_M2_model_weights_2018-09-20-08_55_26_413114.h5',custom_objects={'my_mean_squared_error': my_mean_squared_error, 'my_mean_absolute_error':my_mean_absolute_error})

# In[]: 2. Performance_table_RP
##宣告函數
def generate_performance_table_RP(y_true, y_pred):
    cm_temp = [] #先創建一個LIST
    for i in range( 0,(train_y3.shape)[0]): #每一列(會員)，用實際的Item vs 預測的Item，來計算一個confusion matrix
        cm = confusion_matrix(y_true = y_true[i,:],
                         y_pred = y_pred[i,:] )
        if cm.shape == (2, 2): #防止某些會員Item購買量全為0，若發生此狀況，則該會員的performance不記錄
            cm_temp.append([(cm[1,0] +  cm[1,1]) , cm[1,1], (cm[0,1] +  cm[1,1]) , cm[1,1]])
        if i % 1000 == 0 :
            print(i)
    performance_table_RP = pd.DataFrame(data=cm_temp)
    performance_table_RP.rename(columns={0:'True1', 1:'Predict1_of_true1',
                        2:'Predict1',3:'true1_of_Predict1'},inplace=True)#注意这里0和1都不是字符串
    return performance_table_RP

##train生成預測值
train_y3_01 = np.where(train_y3> 0, 1, 0) #y_true

#以下三個模型選一個來生出預測值:--------------------------
train_y3_hat = model.predict(train_x3) #M1
#train_y3_hat = model.predict([train_x1,train_x2,train_x3]) #M2
#train_y3_hat = model.predict([train_x1,train_x2,train_x3,train_x1_cul,train_x2_cul,train_x3_cul]) #M3

train_y3_hat_inverse_transform = Scaler_train_y3.inverse_transform(train_y3_hat)
train_y3_hat_01 = np.where(train_y3_hat_inverse_transform> 0, 1, 0)
#np.savetxt("train_y3_hat.csv", train_y3_hat, delimiter=",") #輸出CSV

#呼叫函數generate_performance_table_RP，來生成performance_table

M1_performance_table = generate_performance_table_RP(y_true=train_y3_01, y_pred=train_y3_hat_01)

##valid生成預測值
valid_y3_01 = np.where(valid_y3> 0, 1, 0) #y_true

#以下三個模型選一個來生出預測值:--------------------------
valid_y3_hat = model.predict(valid_x3) #M1
#valid_y3_hat = model.predict([valid_x1,valid_x2,valid_x3]) #M2
#valid_y3_hat = model.predict([valid_x1,valid_x2,valid_x3,valid_x1_cul,valid_x2_cul,valid_x3_cul]) #M3

valid_y3_hat_inverse_transform = Scaler_train_y3.inverse_transform(valid_y3_hat)
valid_y3_hat_01 = np.where(valid_y3_hat_inverse_transform> 0, 1, 0)
#np.savetxt("valid_y3_hat.csv", valid_y3_hat, delimiter=",") #輸出CSV

#呼叫函數generate_performance_table_RP，來生成performance_table
M1_performance_table_valid = generate_performance_table_RP(y_true=valid_y3_01, y_pred=valid_y3_hat_01)

# In[]: # 初始跑一次就好                                                         
#M0:Null Modle 
"""train_y3_hat_NUll = train_x3/3 #M0:Null Modle
M1_performance_table_NUll = generate_performance_table_RP(y_true=train_y3_01, y_pred=train_y3_hat_01)
valid_y3_hat_NUll = valid_x3/3 #M0:Null Modle
M1_performance_table_valid_NUll = generate_performance_table_RP(y_true=train_y3_01, y_pred=train_y3_hat_01)"""


# In[] 3. 畫成果圖
#呼叫自定義函數MyScatterplot，來畫出散布圖
print('Train')
an = M1_performance_table_NUll['True1']
bn = M1_performance_table_NUll['Predict1_of_true1']
cn = M1_performance_table_NUll['Predict1']
dn = M1_performance_table_NUll['true1_of_Predict1']
    
a = M1_performance_table['True1']
b = M1_performance_table['Predict1_of_true1']
c = M1_performance_table['Predict1']
d = M1_performance_table['true1_of_Predict1']

Scatterplot_RP( xTrue = a, yPredict = b, xTrue_NUll = an, yPredict_NUll = bn, lower = 0, upper = 500 ) 
Scatterplot_RP( xTrue = c, yPredict = d, xTrue_NUll = cn, yPredict_NUll = dn, lower = 0, upper = 500 ) 

print('Valid')
an = M1_performance_table_valid_NUll['True1']
bn = M1_performance_table_valid_NUll['Predict1_of_true1']
cn = M1_performance_table_valid_NUll['Predict1']
dn = M1_performance_table_valid_NUll['true1_of_Predict1']

a = M1_performance_table_valid['True1']
b = M1_performance_table_valid['Predict1_of_true1']
c = M1_performance_table_valid['Predict1']
d = M1_performance_table_valid['true1_of_Predict1']
Scatterplot_RP( xTrue = a, yPredict = b, xTrue_NUll = an, yPredict_NUll = bn, lower = 0, upper = 500 ) 
Scatterplot_RP( xTrue = c, yPredict = d, xTrue_NUll = cn, yPredict_NUll = dn, lower = 0, upper = 500 ) 
# In[]: 4. Performance_table_CompareWithNull
###Train extra predict
def Generate_Performance_table_CwN(data_yTrue, y3_hat_NUll, y3_hat_01):
    a = np.where( ( data_yTrue > 0 ) & ( y3_hat_NUll == 0 ) , 1, 0)
    b = np.where( ( a == 1 ) & ( y3_hat_01 > 0 ) , 1, 0)
    c = np.where( ( data_yTrue > 0 ) & ( y3_hat_NUll > 0 ) , 1, 0)
    d = np.where( ( c == 1 ) & ( y3_hat_01 > 0 ) , 1, 0)
    cm_temp = [] #先創建一個LIST
    for i in range( 0,(train_y3.shape)[0]): #每一列(會員)，用實際的Item vs 預測的Item，來計算一個confusion matrix
        a_temp = sum( a[i,:] )
        b_temp = sum( b[i,:] )
        c_temp = sum( c[i,:] )
        d_temp = sum( d[i,:] )
        cm_temp.append([a_temp, b_temp, c_temp, d_temp])
        if i % 1000 == 0 :
            print(i)
    M1_performance_table_CompareWithNull = pd.DataFrame(data=cm_temp)
    M1_performance_table_CompareWithNull.rename(columns={0:'Number_not_in_NullModel',
       1:'Number_not_in_NullModel_but_predict', 2:'Number_in_NullModel', 3:'Number_in_NullModel_and_predict'}, inplace=True)#注意这里0和1都不是字符串
    return M1_performance_table_CompareWithNull

M1_performance_table_CompareWithNull = Generate_Performance_table_CwN(train_y3_01, train_y3_hat_NUll, train_y3_hat_01)
M1_performance_table_valid_CompareWithNull = Generate_Performance_table_CwN(valid_y3_01, valid_y3_hat_NUll, valid_y3_hat_01)

# In[]: 5. 畫成果圖 Compare With NullModel
# Compare With NullModel
print('Train Extra Predict')
a = M1_performance_table_CompareWithNull['Number_not_in_NullModel']
b = M1_performance_table_CompareWithNull['Number_not_in_NullModel_but_predict']
Scatterplot_CompareWithNull( xTrue = a, yPredict = b, xlabel=" #Item not in Null Model ", ylabel = " #(Item not in Null Model but predicted) ", lower = 0, upper = 200 ) 
a = M1_performance_table_CompareWithNull['Number_in_NullModel']
b = M1_performance_table_CompareWithNull['Number_in_NullModel_and_predict']
Scatterplot_CompareWithNull( xTrue = a, yPredict = b, xlabel=" #Item in Null Model ", ylabel = " #(Item in Null Model and predicted) ", lower = 0, upper = 200 ) 

print('Valid Extra Predict')
a = M1_performance_table_valid_CompareWithNull['Number_not_in_NullModel']
b = M1_performance_table_valid_CompareWithNull['Number_not_in_NullModel_but_predict']
Scatterplot_CompareWithNull( xTrue = a, yPredict = b, xlabel=" #Item not in Null Model ", ylabel = " #(Item not in Null Model but predicted) ", lower = 0, upper = 200 ) 
a = M1_performance_table_valid_CompareWithNull['Number_in_NullModel']
b = M1_performance_table_valid_CompareWithNull['Number_in_NullModel_and_predict']
Scatterplot_CompareWithNull( xTrue = a, yPredict = b, xlabel=" #Item in Null Model ", ylabel = " #(Item in Null Model and predicted) ", lower = 0, upper = 200 ) 

# In[]: 6. 直方圖
r=b/a #recall
r=r.loc[~r.isna()]
p=d/c
p=p.loc[~p.isna()]
sns.distplot( r, bins=None) #recall
sns.distplot( p, bins=None) #precision
beta = 0.10 #beta = P/R的看重程度
f1 = ((1+beta**2)*(p*r))/( ( beta**2)*p + r )
sns.distplot(f1.loc[~f1.isna()], bins=None) # f1
(f1.loc[~f1.isna()]).describe() #敘述統計
# In[]:plot model architecture
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

# In[] M4 data : current Q + Cumulation + 工作日\假日 :

list_temp = ["M1WE","M2WE","M3WE","M4WE","M5WE","M6WE","M7WE","M8WE","M9WE","M10WE", \
            "M1WD","M2WD","M3WD","M4WD","M5WD", "M6WD","M7WD","M8WD","M9WD","M10WD"] 

train_x1_Q_WD, Clipper_train_x1_Q_WD, Scaler_train_x1_Q_WD = Data_Aggregate_Clip_Scal(dataset_name_list = ["M7WD","M8WD","M9WD"], Dataset = Item1,  Clipper=None, Scaler=None)
train_x1_Q_WE, Clipper_train_x1_Q_WE, Scaler_train_x1_Q_WE = Data_Aggregate_Clip_Scal(dataset_name_list = ["M7WE","M8WE","M9WE"], Dataset = Item1,  Clipper=None, Scaler=None)
train_x1_cul_WD, Clipper_train_x1_cul_WD, Scaler_train_x1_cul_WD = Data_Aggregate_Clip_Scal(dataset_name_list = ["M1WD","M2WD","M3WD","M4WD","M5WD", "M6WD","M7WD","M8WD","M9WD","M10WD"], Dataset = Item1,  Clipper=None, Scaler=None)
train_x1_cul_WE, Clipper_train_x1_cul_WE, Scaler_train_x1_cul_WE = Data_Aggregate_Clip_Scal(dataset_name_list = ["M1WE","M2WE","M3WE","M4WE","M5WE","M6WE","M7WE","M8WE","M9WE","M10WE"], Dataset = Item1,  Clipper=None, Scaler=None)
train_x2_Q_WD, Clipper_train_x2_Q_WD, Scaler_train_x2_Q_WD = Data_Aggregate_Clip_Scal(dataset_name_list = ["M7WD","M8WD","M9WD"], Dataset = Item2,  Clipper=None, Scaler=None)
train_x2_Q_WE, Clipper_train_x2_Q_WE, Scaler_train_x2_Q_WE = Data_Aggregate_Clip_Scal(dataset_name_list = ["M7WE","M8WE","M9WE"], Dataset = Item2,  Clipper=None, Scaler=None)
train_x2_cul_WD, Clipper_train_x2_cul_WD, Scaler_train_x2_cul_WD = Data_Aggregate_Clip_Scal(dataset_name_list = ["M1WD","M2WD","M3WD","M4WD","M5WD", "M6WD","M7WD","M8WD","M9WD","M10WD"], Dataset = Item2,  Clipper=None, Scaler=None)
train_x2_cul_WE, Clipper_train_x2_cul_WE, Scaler_train_x2_cul_WE = Data_Aggregate_Clip_Scal(dataset_name_list = ["M1WE","M2WE","M3WE","M4WE","M5WE","M6WE","M7WE","M8WE","M9WE","M10WE"], Dataset = Item2,  Clipper=None, Scaler=None)
train_x3_Q_WD, Clipper_train_x3_Q_WD, Scaler_train_x3_Q_WD = Data_Aggregate_Clip_Scal(dataset_name_list = ["M7WD","M8WD","M9WD"], Dataset = Item3,  Clipper=None, Scaler=None)
train_x3_Q_WE, Clipper_train_x3_Q_WE, Scaler_train_x3_Q_WE = Data_Aggregate_Clip_Scal(dataset_name_list = ["M7WE","M8WE","M9WE"], Dataset = Item3,  Clipper=None, Scaler=None)
train_x3_cul_WD, Clipper_train_x3_cul_WD, Scaler_train_x3_cul_WD = Data_Aggregate_Clip_Scal(dataset_name_list = ["M1WD","M2WD","M3WD","M4WD","M5WD", "M6WD","M7WD","M8WD","M9WD","M10WD"], Dataset = Item3,  Clipper=None, Scaler=None)
train_x3_cul_WE, Clipper_train_x3_cul_WE, Scaler_train_x3_cul_WE = Data_Aggregate_Clip_Scal(dataset_name_list = ["M1WE","M2WE","M3WE","M4WE","M5WE","M6WE","M7WE","M8WE","M9WE","M10WE"], Dataset = Item3,  Clipper=None, Scaler=None)
                    
#以下
 
# In[]:M4
A1 = Input(shape = (99,),name='Input_1')
A2 = Input(shape = (424,),name='Input_2')
A3 = Input(shape = (2404,),name='Input_3')
A1_cul = Input(shape = (99,),name='Input_1_cul')
A2_cul = Input(shape = (424,),name='Input_2_cul')
A3_cul = Input(shape = (2404,),name='Input_3_cul')
Merged = concatenate([A1, A2, A3, A1_cul, A2_cul, A3_cul ])
B = Dense(2000,activation='selu')(Merged)
B = Dropout(0.4)(B)
C = Dense(1000,activation='selu',)(B)
C = Dropout(0.4)(C)
D = Dense(500,activation='relu',)(C)
D = Dropout(0.4)(D)
N = Dense(200,activation='relu',)(D)
N = Dropout(0.4)(N)
O = Dense(500,activation='relu',)(N)
O = Dropout(0.4)(O)
P = Dense(1000,activation='relu',)(O)
P = Dropout(0.4)(P)
y = Dense(2404, activation='relu',name='Output')(P)

model = Model(inputs=[A1, A2, A3, A1_cul, A2_cul, A3_cul ], outputs=[y])
model.compile(loss = my_mean_squared_error, optimizer =  adam, metrics=[my_mean_absolute_error]) #mean_absolute_error
model.summary()
modelcallbacks = model.fit(x = [train_x1,train_x2,train_x3,train_x1_cul,train_x2_cul,train_x3_cul], y = train_y3,
                epochs=5, batch_size=64, shuffle=True,
                validation_data=([valid_x1,valid_x2,valid_x3,valid_x1_cul,valid_x2_cul,valid_x3_cul], valid_y3,))

model.save(  'M2_model_weights_'+ str(datetime.datetime.now() ).replace(" ", "-").replace(":", "_").replace(".", "_")+'.h5'  )
#model=load_model('save_M2_model_weights_2018-09-20-08_55_26_413114.h5',custom_objects={'my_mean_squared_error': my_mean_squared_error, 'my_mean_absolute_error':my_mean_absolute_error})
# In[]: 1. Summarize history for loss  
plt.figure()
plt.plot(modelcallbacks.history['loss'])  #loss mean_absolute_error 
plt.plot(modelcallbacks.history['val_loss'])  #val_loss val_mean_absolute_error
plt.plot(modelcallbacks.history['my_mean_absolute_error'])  #loss mean_absolute_error 
plt.plot(modelcallbacks.history['val_my_mean_absolute_error'])  #val_loss val_mean_absolute_error
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error'], loc='upper left')
plt.show()
# In[]:M1 4HL relu +　mean_squared_error
A1 = Input(shape = (2404,),name='Input')
B = Dense(800,activation='relu')(A1)
B = Dropout(0.4)(B)
C = Dense(250,activation='relu',)(B)
C = Dropout(0.4)(C)
N = Dense(80,activation='relu',)(C)
N = Dropout(0.4)(N)
O = Dense(250,activation='relu',)(N)
O = Dropout(0.4)(O)
P = Dense(800,activation='relu',)(O)
P = Dropout(0.4)(P)
y = Dense(2404, activation='relu',name='Output')(P)
# In[]:M1 sigmoid + binary_crossentropy

#M1 Input Data 將值轉換到0,1之間
train_x1_01 = np.where(train_x1> 0, 1, 0)
train_y3_01 = np.where(train_y3> 0, 1, 0)
valid_x1_01 = np.where(valid_x1> 0, 1, 0)
valid_y3_01 = np.where(valid_y3> 0, 1, 0)

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
y = Dense(2404, activation='sigmoid',name='Output')(P)

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
modelcallbacks = model.fit(x = [train_x1_WE,train_x1_WD,train_x2_WE,train_x2_WD], y = train_y3,
                epochs=15, batch_size=64, shuffle=True,
                validation_data=([valid_x1_WE,valid_x1_WD,valid_x2_WE,valid_x2_WD], valid_y3,))


# Test data performance
#testPerformance = 
"""model.evaluate([train_x1_WE,train_x1_WD,train_x2_WE,train_x2_WD], valid_y3,
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
