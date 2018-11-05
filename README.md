# Peer-Based-Deep-Learning
Peer-based deep learning: what retail customers will buy and how to create customized product sets

## 1.資料夾Potential_Question_YisFreq_0821：將下一個月會員來店的次數當作預測變數(Y)
### 1.1使用以下三類描述變數：X1:會員經過計算的資訊(RFMC、年齡、性別......等等)、X2：當季購買數量總和、X3：累積購買數量總和


## 2.資料夾Potential_Question2_YisItems：將下一個月會員會購買的單品當作預測變數(Y)
### 2.1使用以下四類描述變數：X1:當季購買單品數量總和、X2：X1累積資料、X3：當季購買品號或是群號數量總和、X4：X3累積資料
### 2.2簡化版模型架構： 
![](https://github.com/YuTaNCCU/Peer-Based-Deep-Learning/blob/master/Potential_Question2_YisItems/model%20architecture.PNG)
### 2.3模型細節：M0:X為前一季有買過的單品、M1:X1、M2:X1+X2、M3:X1+X2+X3
### 2.4模型結果(M1、M2、M3)在Train Set(下圖左)及Valid Set(下圖右)都較Null Model(M0)預測出更多會員會買的商品： 
![](https://github.com/YuTaNCCU/Peer-Based-Deep-Learning/blob/master/Potential_Question2_YisItems/M0123compare.PNG)
