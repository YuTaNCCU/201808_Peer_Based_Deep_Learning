
# coding: utf-8

# In[1]:Import packages


import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from IPython.display import display, clear_output
import datetime
from IPython.core.interactiveshell import InteractiveShell  #display all line
InteractiveShell.ast_node_interactivity = "all"



# In[] Loading Data


# Loading Data
#Q1_Item = pd.read_csv('data/Q1_Item.csv', encoding = 'big5hkscs' )
#Q2_Item = pd.read_csv('data/Q2_Item.csv', encoding = 'big5hkscs' )
#All_Item = pd.read_csv('data/All_Item.csv', encoding = 'big5hkscs' )
AllMb = pd.read_csv('data/ALLFAM08_AllMb.csv', encoding = 'big5hkscs' )
OriginData1to12 = pd.read_csv('data/OriginData1to12.csv', encoding = 'big5hkscs' )
OriginData1to12_SecondNumber = pd.read_csv('data/OriginData1to12_SecondNumber.csv' )
All_Item2000 = pd.read_csv('data/All_Item2000.csv' )

# In[]:  刪掉部分不活躍的會員 : 創建各季個別來幾次的DataFrame


#創建各季各來了幾次
temp = OriginData1to12[["會員卡號", "交易id", "資料日期", "季分"]].groupby(["交易id","會員卡號"]).first() 
temp = temp.reset_index(level="交易id", drop=True)
temp2=temp.iloc[:,].pivot_table(index=['會員卡號'], columns = ['季分'],values = ['季分'], 
                                aggfunc = lambda x: sum(1 for y in x ) )  #使用pviot
temp2.columns=[['Q1','Q2','Q3','Q4']]
#temp2.fillna( 0 ,inplace = True)
temp2.head()


# In[]:

temp3 = temp2.iloc[:,:].apply(lambda x:  x.isna().sum() ,axis=1) 
temp3 = pd.DataFrame(temp3)
temp3.reset_index(level="會員卡號", drop=False, inplace=True)
temp3.columns=[['會員卡號','Na']]
temp3.columns = temp3.columns.get_level_values(0) #flatten a hierarchical index in columns
SelectefMb = temp3.loc[ temp3.Na<=2 , "會員卡號"]#至少出現過2季：57159人,3季:44249人,4季:31893人

OriginData1to12_SelectedMb = OriginData1to12.loc[OriginData1to12.會員卡號.isin(SelectefMb) ,] 

# In[37]:



# 預處理大表
OriginData1to12_SecondNumber = OriginData1to12.copy()

# 將品號-品名稱分開 ＋  將群號-群名稱分開
OriginData1to12_SecondNumber["FirstNumber"] = OriginData1to12.loc[:,"品號-品名稱"].str.split('-', expand = True).iloc[:,0]
OriginData1to12_SecondNumber["SecondNumber"] = OriginData1to12.loc[:,"群號-群名稱"].str.split('-', expand = True).iloc[:,0]
#OriginData1to12_SecondNumber = OriginData1to12_SecondNumber[["FirstNumber", "SecondNumber","單品名稱", "銷售數量"]].copy()

# 有空值者，補成字元型態“NaN”
OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["FirstNumber"].isna(), "FirstNumber"] = "NaN"
OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["SecondNumber"].isna(), "SecondNumber"] = "NaN"
OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["單品名稱"].isna(), "單品名稱"] = "NaN"

#輸出
OriginData1to12_SecondNumber.to_csv("data/OriginData1to12_SecondNumber.csv",index=False)
OriginData1to12_SecondNumber.head()


# In[119]:


#使用單品名稱 來group_by 及配對，並自訂max_features數目
def tokenize_AllItem(Qdata, max_features):
    """使用Qdata中的商品購買次數來排序出前面max_features個最常購買商品
    """
    max_features = max_features

    #預處理All_Item
    All_Item = pd.DataFrame()#準備一個新的DataFrame，放所有商品對應單品名稱的編號
    All_Item["ItemCount"] = Qdata.groupby("單品名稱").apply(lambda Qdata:sum(Qdata["銷售數量"]))
    All_Item["SecondNumber"] = Qdata[["單品名稱", "SecondNumber"]].groupby("單品名稱").first() #選擇第一個群號來當該單品平稱的群號
    All_Item = All_Item.reset_index(level="單品名稱", drop=False)
    All_Item = All_Item.sort_values(by=["ItemCount"], ascending= False ).reset_index(drop=True)  #依 ItemToken 排序（此時ItemToken代表sum("銷售數量")）
    All_Item.loc[All_Item["單品名稱"] == "NaN","SecondNumber"] =  "1000"  #單品名稱為NaN者，給定群號1000  （原始群號最大為988）

    # 先把所有ItemToken欄位都變成固定值
    ## 法一：All_Item Tokenize： 都變成對應的群號
    All_Item["ItemToken"] =  "1000" + All_Item.loc[:,"SecondNumber"]
    """## 法二：All_Item Tokenize： 先把所有ItemToken欄位都變成0
    All_Item["ItemToken"] = 0"""

    # 再將前max_features的單品名稱給予編號 ItemToken
    for i in range(1, max_features+1):
        All_Item.loc[i-1,"ItemToken" ]= str( i )

    # 轉換成整數
    All_Item["ItemToken"] = All_Item["ItemToken"].astype(int)
    
    # 保留 ItemToken = 0 這個結果，用作單品名稱未見及群號未出現的情況
    DataFrame_temp = pd.DataFrame(data={'單品名稱': ['單品名稱未見及群號未出現'], 
                                        'ItemCount':[999999], 'SecondNumber':[999],  'ItemToken' : [0] })
    All_Item = pd.concat([DataFrame_temp, All_Item], axis = 0 ,sort=True).reset_index(drop=True)

    return All_Item

All_Item8000 = tokenize_AllItem(Qdata = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q1",],
                                max_features = 8000)

#寫入檔案
All_Item8000.to_csv("data/All_Item8000.csv",index=False)
#All_Item = pd.read_csv('data/All_Item2000.csv' )


# In[128]:


#所有會員購買的單品名稱 to Item > 再轉換為寬表格
def tokenize_MbRecord(Qdata_Item, All_Item_maxfeatures):
    """
    >將包含購買資料的大資料表，轉換為以「MbID」為單位的寬表格(稀疏表格)
    >Args:
    Qdata_Item:要轉換的、包含購買資料的、某一季大資料表
    All_Item_maxfeatures:包含單品名稱、ItemToken的資料表，且已經決定maxfeatures要取多少
    >Returns:A DataFrame
    """
    Qdata_Item = Qdata_Item.copy() #複製DataFrame
    Qdata_Item.loc[Qdata_Item["單品名稱"].isna(), "單品名稱" ]  = "NaN" # 有空值者，補成字元型態“NaN”
    #------------------------------------------------------------------------------------------------#
    # texts_to_sequences : 將Q_Item的 "單品名稱"欄位 轉換成 "item"欄位
    All_Item_maxfeatures_temp = All_Item_maxfeatures[['單品名稱','ItemToken']].set_index('單品名稱') #將單品名稱設定成index，以利搜尋
    All_Item_maxfeatures_temp2 = All_Item_maxfeatures[["SecondNumber",'ItemToken']].set_index('SecondNumber') #將SecondNumber設定成index，以利搜尋
    List_temp= []#創建一個List
    start = datetime.datetime.now() #計時用
    #轉換
    for j in range(len( Qdata_Item )):  
        str_temp = Qdata_Item.loc[j, "單品名稱" ]
        str_temp2 = Qdata_Item.loc[j, "SecondNumber" ]
        if All_Item_maxfeatures_temp.index.contains( str_temp ) :
            List_temp.append( int (  All_Item_maxfeatures_temp.loc[ str_temp ].values  ))
        elif All_Item_maxfeatures_temp2.index.contains( str_temp2 ) :
            List_temp.append( int ( "1000" + str_temp2  ))
        else :
            List_temp.append( 0 )
        if j%1000 == 0:  #顯示進度
            clear_output()
            print( j, datetime.datetime.now() )

    Qdata_Item['ItemToken'] = List_temp #把List加入Dataframe
    Qdata_Item.head(100)
    #Qdata_Item.to_csv("data/Qdata_Item.csv",index=False)
    #------------------------------------------------------------------------------------------------#
    # 長表格 to 寬表格
    Qdata_Item_sparse=Qdata_Item.iloc[:,].pivot_table(index=['會員卡號'], columns = ['ItemToken'], values = ['銷售數量'], aggfunc = lambda x: np.sum(x) )  #使用pviot
    #Qdata_Item_sparse = pd.crosstab(Qdata_Item.loc[:,'會員卡號'], Qdata_Item.loc[:,'ItemToken'] , margins=False)

    # pivot_table to DataFrame
    Qdata_Item_sparse = pd.DataFrame(Qdata_Item_sparse.to_records())

    # 將欄位名稱簡化
    Qdata_Item_sparse.rename(columns={'會員卡號':'MbID'}, inplace=True)
    Qdata_Item_sparse.columns = [i.replace("('銷售數量', ", "").replace(")", "") for i in Qdata_Item_sparse.columns]
    #------------------------------------------------------------------------------------------------#
    # 未使用到的單品名稱，仍要有欄位
    # .1找出沒使用到的ItemToken
    NoShowItem = All_Item_maxfeatures.loc[~ All_Item_maxfeatures['ItemToken'].isin(Qdata_Item['ItemToken']), "ItemToken"].unique()

    # .2創建包含沒使用到的ItemToken作為欄位的空np陣列
    df_temp = pd.DataFrame(
        data = np.zeros(NoShowItem.shape[0]*Qdata_Item_sparse.shape[0]).reshape(Qdata_Item_sparse.shape[0],NoShowItem.shape[0]),
                      columns=NoShowItem)
    Qdata_Item_sparse = pd.concat([Qdata_Item_sparse, df_temp], axis =1 ) #合併DataFrame(左)與np陣列(右)
    # .3排序欄位名稱append
    Qdata_Item_sparse2 =  Qdata_Item_sparse.drop(['MbID'], axis=1).copy() #先去掉欄位MbID
    Qdata_Item_sparse2.columns = [int(i) for i in Qdata_Item_sparse2.columns] #欄位轉成整數
    Qdata_Item_sparse2.sort_index(axis=1, inplace=True) #排序欄位名稱
    Qdata_Item_sparse = pd.concat([Qdata_Item_sparse['MbID'].astype(int) ,Qdata_Item_sparse2], axis=1) #加回欄位MbID
    #------------------------------------------------------------------------------------------------#
    # Customers who didn't show in the quarter
    NoShowMb = AllMb.loc[  ~ AllMb.MbID.isin(Qdata_Item_sparse['MbID'])] # 該季未出現的會員Dataframe
    temp = pd.DataFrame(data={'MbID':NoShowMb["MbID"]}) # 該季未出現的會員含所有欄位的sparse Dataframe
    Qdata_Item_sparse = pd.concat([Qdata_Item_sparse, temp], 
                                  axis = 0,sort=False).sort_values(by=["MbID"]).reset_index(drop=True) #合併成一表
    # NaN補0
    Qdata_Item_sparse.fillna(value = 0, inplace=True)
    #------------------------------------------------------------------------------------------------#
    # normalize  
    temp = Qdata_Item_sparse.copy().drop(['MbID'], axis=1)
    temp = (temp - temp.min()) /(temp.max() - temp.min())
    Qdata_Item_sparse = pd.concat([Qdata_Item_sparse['MbID'].astype(int) ,temp], axis=1)
    # NaN補0，以免出現0/0=NaN
    Qdata_Item_sparse.fillna(value = 0, inplace=True)
    #------------------------------------------------------------------------------------------------#
    display(
        Qdata_Item_sparse.head())
    return (Qdata_Item_sparse)


# In[129]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q1",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item2000 )
result_temp.to_csv("data/Q1_Item_2000.csv",index=False)


# In[130]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q2",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item2000 )
result_temp.to_csv("data/Q2_Item_2000.csv",index=False)


# In[131]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q3",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item2000 )
result_temp.to_csv("data/Q3_Item_2000.csv",index=False)


# In[132]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q1",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item5000 )
result_temp.to_csv("data/Q1_Item_5000.csv",index=False)


# In[133]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q2",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item5000 )
result_temp.to_csv("data/Q2_Item_5000.csv",index=False)


# In[134]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q3",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item5000 )
result_temp.to_csv("data/Q3_Item_5000.csv",index=False)


# In[135]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q1",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item8000 )
result_temp.to_csv("data/Q1_Item_8000.csv",index=False)


# In[136]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q2",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item8000 )
result_temp.to_csv("data/Q2_Item_8000.csv",index=False)


# In[137]:


data_temp = OriginData1to12_SecondNumber.loc[OriginData1to12_SecondNumber["季分"] == "Q3",].copy().reset_index(drop=True) #複製DataFrame
data_temp = data_temp.iloc[:,].copy()
result_temp = tokenize_MbRecord(Qdata_Item = data_temp, All_Item_maxfeatures = All_Item8000 )
result_temp.to_csv("data/Q3_Item_8000.csv",index=False)

