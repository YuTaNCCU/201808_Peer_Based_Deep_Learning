
# coding: utf-8

# In[]:Import packages
import pandas as pd
import numpy as np
from IPython.display import display, clear_output
import datetime
from IPython.core.interactiveshell import InteractiveShell  #display all line
InteractiveShell.ast_node_interactivity = "all"
# In[] Loading Data

OriginData1to12 = pd.read_csv('data/OriginData1to12.csv', encoding = 'big5hkscs' )
OriginData1to12_SelectedMb = pd.read_csv('data/OriginData1to12_SelectedMb.csv' )
SelectedMb = pd.read_csv('data/SelectedMb.csv', encoding = 'big5hkscs' )
All_Item2000 = pd.read_csv('data/All_Item2000.csv' )
OriginData_Processed = pd.read_csv('data/OriginData_Processed.csv' )

# In[]:  刪掉部分不活躍的會員 : 創建各季個別來幾次的DataFrame

#創建各季各來了幾次
temp = OriginData1to12[["會員卡號", "交易id", "資料日期", "季分"]].groupby(["交易id","會員卡號"]).first() 
temp = temp.reset_index(level="交易id", drop=True)
#pivot_table
temp2=temp.iloc[:,].pivot_table(index=['會員卡號'], columns = ['季分'],values = ['季分'], 
                                aggfunc = lambda x: sum(1 for y in x ) )  #使用pviot
temp2.columns=[['Q1','Q2','Q3','Q4']]
#temp2.fillna( 0 ,inplace = True)
temp2.head()
#計算有幾季未出現
temp3 = temp2.iloc[:,:].apply(lambda x:  x.isna().sum() ,axis=1) 
temp3 = pd.DataFrame(temp3)
temp3.reset_index(level="會員卡號", drop=False, inplace=True)
temp3.columns=[['會員卡號','Na']]
temp3.columns = temp3.columns.get_level_values(0) #flatten a hierarchical index in columns
# 選擇至少出現過2季
SelectedMb = temp3.loc[ temp3.Na<=0 , "會員卡號"]#至少出現過2季：57159人,3季:44249人,4季:31893人
SelectedMb.to_csv("data/SelectedMb.csv",index=False)
#篩選出僅含SelectefMb的大表
OriginData1to12_SelectedMb = OriginData1to12.loc[OriginData1to12.會員卡號.isin(SelectedMb) ,] 
OriginData1to12_SelectedMb.to_csv("data/OriginData1to12_SelectedMb.csv",index=False)

# In[]: 預處理大表
OriginData_Processed = OriginData1to12_SelectedMb.copy()

# 將品號-品名稱分開 ＋  將群號-群名稱分開
OriginData_Processed["FirstNumber"] = OriginData1to12_SelectedMb.loc[:,"品號-品名稱"].str.split('-', expand = True).iloc[:,0]
OriginData_Processed["SecondNumber"] = OriginData1to12_SelectedMb.loc[:,"群號-群名稱"].str.split('-', expand = True).iloc[:,0]

# 有空值者，補成字元型態“NaN”
OriginData_Processed.loc[OriginData_Processed["FirstNumber"].isna(), "FirstNumber"] = "空值"
OriginData_Processed.loc[OriginData_Processed["SecondNumber"].isna(), "SecondNumber"] = "空值"
OriginData_Processed.loc[OriginData_Processed["單品名稱"].isna(), "單品名稱"] = "空值"

#將日期轉換為星期幾:0:週一、...、6:週日
OriginData_Processed['day_of_week'] = pd.to_datetime(OriginData_Processed.loc[:,"資料日期"]).dt.dayofweek
#輸出月份
OriginData_Processed["Month"] = pd.to_datetime(OriginData_Processed.loc[:,"資料日期"]).dt.month

#輸出
OriginData_Processed.to_csv("data/OriginData_Processed.csv",index=False)
OriginData_Processed.head()

# In[]: 使用單品名稱 來group_by 及配對，並自訂max_features數目
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
    All_Item.loc[All_Item["單品名稱"] == "空值","SecondNumber"] =  "1000"  #單品名稱為NaN者，給定群號1000  （原始群號最大為988）

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



All_Item2000 = tokenize_AllItem(Qdata = OriginData_Processed.loc[OriginData_Processed.資料日期 <= "2017-10-31",:],
                                max_features = 2000) #2404種
All_Item2000.to_csv("data/All_Item2000.csv",index=False)#寫入檔案

All_Item5000 = tokenize_AllItem(Qdata = OriginData_Processed.loc[OriginData_Processed.資料日期 <= "2017-10-31",:],
                                max_features = 5000) #5343種
All_Item5000.to_csv("data/All_Item5000.csv",index=False)#寫入檔案

All_Item8000 = tokenize_AllItem(Qdata = OriginData_Processed.loc[OriginData_Processed.資料日期 <= "2017-10-31",:],
                                max_features = 8000) #8293種
All_Item8000.to_csv("data/All_Item8000.csv",index=False)#寫入檔案
#All_Item8000 = pd.read_csv('data/All_Item8000.csv' )


# In[]: 所有會員購買的單品名稱 to token > 再轉換為寬表格
def tokenize_MbRecord(FilteredData, All_Item_maxfeatures, maxfeatures):
    """
    >將包含購買資料的大資料表，轉換為以「MbID」為單位的寬表格(稀疏表格)
    >Args:
    Qdata_Item:要轉換的、包含購買資料的、某一季大資料表
    All_Item_maxfeatures:包含單品名稱、ItemToken的資料表，且已經決定maxfeatures要取多少
    maxfeatures:清楚說明maxfeatures要取多少,其值會用在步驟:#單品名稱或SecondNumber轉換成ItemToken
    >Returns:A DataFrame
    """
    Qdata_Item = FilteredData.copy() #複製DataFrame
    Qdata_Item.loc[Qdata_Item["單品名稱"].isna(), "單品名稱" ]  = "空值" # 有空值者，補成字元型態“NaN”
    #------------------------------------------------------------------------------------------------#
    # texts_to_sequences : 將Q_Item的 "單品名稱"欄位 轉換成 "item"欄位
    All_Item_maxfeatures_temp = All_Item_maxfeatures[['單品名稱','ItemToken']].set_index('單品名稱') #將單品名稱設定成index，以利搜尋
    All_Item_maxfeatures_temp2 = All_Item_maxfeatures[["SecondNumber",'ItemToken']].iloc[maxfeatures+1:,:].set_index('SecondNumber') #將SecondNumber設定成index，以利搜尋
    List_temp= []#創建一個List
    start = datetime.datetime.now() #計時用
    #單品名稱或SecondNumber轉換成ItemToken
    for j in range(len( Qdata_Item )):  
        str_temp = Qdata_Item.loc[j, "單品名稱" ]
        str_temp2 =  Qdata_Item.loc[j, "SecondNumber" ]
        if All_Item_maxfeatures_temp.index.contains( str_temp ) :
            List_temp.append( int (  All_Item_maxfeatures_temp.loc[ str_temp ].values  ))
        elif All_Item_maxfeatures_temp2.index.contains( str_temp2 ) :
            List_temp.append(  int ( "1000" + str_temp2  ) )
        else :
            List_temp.append( 0 )
        if j%10000 == 0:  #顯示進度
            clear_output()
            print( j, "/" , len( Qdata_Item ) , datetime.datetime.now() )

    Qdata_Item['ItemToken'] = List_temp #把List加入Dataframe
    #Qdata_Item.head(100)
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
    NoShowMb = SelectedMb[  ~ SelectedMb.isin(Qdata_Item_sparse.loc[:,'MbID'])] # 該季未出現的會員Dataframe
    temp = pd.DataFrame(data={'MbID':NoShowMb}) # 該季未出現的會員含所有欄位的sparse Dataframe
    Qdata_Item_sparse = pd.concat([Qdata_Item_sparse, temp], 
                                  axis = 0,sort=False).sort_values(by=["MbID"]).reset_index(drop=True) #合併成一表
    # NaN補0
    Qdata_Item_sparse.fillna(value = 0, inplace=True)
    #------------------------------------------------------------------------------------------------#
    # normalize   #不scaling, 等加總後\要放進模型前，再不scaling
    """temp = Qdata_Item_sparse.copy().drop(['MbID'], axis=1)
    temp = (temp - temp.min()) /(temp.max() - temp.min())
    Qdata_Item_sparse = pd.concat([Qdata_Item_sparse['MbID'].astype(int) ,temp], axis=1)
    Qdata_Item_sparse.fillna(value = 0, inplace=True)# NaN補0，因為0/0會等於NaN"""
    #------------------------------------------------------------------------------------------------#
    display(
        Qdata_Item_sparse.head())
    return (Qdata_Item_sparse)


# In[]: 呼叫tokenize_MbRecord，生成各個月份的 Mb Item Sparse Table
MbItemSparseTable = {} #創建一個DICTIONARY
for month in range(1,13):
    for weekday in ["WD","WE"]:
        
        if weekday == "WD" :
            filter = (OriginData_Processed.Month == month)&(OriginData_Processed.day_of_week.isin([0,1,2,3,4])) 
        else :
            filter = (OriginData_Processed.Month == month)&(OriginData_Processed.day_of_week.isin([5,6])) 
        data_temp = OriginData_Processed.loc[filter,:].reset_index(drop=True).iloc[:,].copy() #篩選資料並複製DataFrame
        
        str_temp = "M" + str(month) + str(weekday) #創建DICTIONARY 的KEY
        print("#-----------------------------------------------",str_temp,"-----------------------------------------#")
        MbItemSparseTable[str_temp]  = tokenize_MbRecord(FilteredData = data_temp,
                         All_Item_maxfeatures = All_Item2000, maxfeatures = 2000 ) #呼叫tokenize_MbRecord，並把回傳的DATAFRAME存在DICTIONARY
        MbItemSparseTable[str_temp].to_csv(str_temp + ".csv") #個別存成csv

# In[]: Load Data
#MbItemSparseTable = {} 
Item = {} 
for month in range(1,3):
   for weekday in ["WD","WE"]:
       str_temp = "M" + str(month) + str(weekday) #創建DICTIONARY 的KEY    
       Item[str_temp] = pd.read_csv(str_temp + ".csv") #個別存成csv
b = Item["M5WE"]+ Item["M5WE"]+ Item["M5WE"]+ Item["M5WE"]+ Item["M5WE"]+ \
    Item["M5WE"]+ Item["M5WE"]+ Item["M5WE"]+ Item["M5WE"]+ Item["M5WE"]
 
 
 
