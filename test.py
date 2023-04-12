# 請勿更動此區塊程式碼

import time
import numpy as np
import pandas as pd

EXECUTION_START_TIME = time.time() # 計算執行時間

df = pd.read_csv('train.csv')      # 讀取資料，請勿更改路徑

# 資料分析與前處理

train_x = df[['Sex', 'Age']]                   # 取出訓練資料需要分析的資料欄位
train_y = df['Survived']                       # 取出訓練資料的答案

from sklearn.impute import SimpleImputer       # 匯入填補缺失值的工具
from sklearn.preprocessing import LabelEncoder # 匯入 Label Encoder

imputer = SimpleImputer(strategy='median')     # 創造 imputer 並設定填補策略
age = train_x['Age'].to_numpy().reshape(-1, 1)
imputer.fit(age)                               # 根據資料學習需要填補的值
train_x['Age'] = imputer.transform(age)        # 填補缺失值

le = LabelEncoder()                            # 創造 Label Encoder
le.fit(train_x['Sex'])                         # 給予每個類別一個數值
train_x['Sex'] = le.transform(train_x['Sex'])  # 轉換所有類別成為數值

# 模型訓練

from sklearn.model_selection import KFold             # 匯入 K 次交叉驗證工具
from sklearn.tree import DecisionTreeClassifier       # 匯入決策樹模型
from sklearn.metrics import accuracy_score            # 匯入準確度計算工具

kf = KFold(n_splits=5,                                # 設定 K 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(train_x):    # 每個迴圈都會產生不同部份的資料
    train_x_split = train_x.iloc[train_index]         # 產生訓練資料
    train_y_split = train_y.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = train_x.iloc[valid_index]         # 產生驗證資料
    valid_y_split = train_y.iloc[valid_index]         # 產生驗證資料標籤
    
    model = DecisionTreeClassifier(random_state=1012) # 創造決策樹模型
    model.fit(train_x_split, train_y_split)           # 訓練決策樹模型
    
    train_pred_y = model.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = model.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
    'average train accuracy: {}\n' +
    '    min train accuracy: {}\n' +
    '    max train accuracy: {}\n' +
    'average valid accuracy: {}\n' +
    '    min valid accuracy: {}\n' +
    '    max valid accuracy: {}').format(
    np.mean(train_acc_list),                          # 輸出平均訓練準確度
    np.min(train_acc_list),                           # 輸出最低訓練準確度
    np.max(train_acc_list),                           # 輸出最高訓練準確度
    np.mean(valid_acc_list),                          # 輸出平均驗證準確度
    np.min(valid_acc_list),                           # 輸出最低驗證準確度
    np.max(valid_acc_list)                            # 輸出最高驗證準確度
))

df = pd.read_csv('test.csv')      # 讀取資料，請勿更改路徑

# 資料分析與前處理

test_x = df[['Sex', 'Age']]                   # 取出訓練資料需要分析的資料欄位

from sklearn.impute import SimpleImputer       # 匯入填補缺失值的工具
from sklearn.preprocessing import LabelEncoder # 匯入 Label Encoder

imputer = SimpleImputer(strategy='median')     # 創造 imputer 並設定填補策略
age = test_x['Age'].to_numpy().reshape(-1, 1)
imputer.fit(age)                               # 根據資料學習需要填補的值
test_x['Age'] = imputer.transform(age)        # 填補缺失值

le = LabelEncoder()                            # 創造 Label Encoder
le.fit(test_x['Sex'])                         # 給予每個類別一個數值
test_x['Sex'] = le.transform(test_x['Sex'])  # 轉換所有類別成為數值


# 模型訓練

from sklearn.model_selection import KFold             # 匯入 K 次交叉驗證工具
from sklearn.tree import DecisionTreeClassifier       # 匯入決策樹模型
from sklearn.metrics import accuracy_score            # 匯入準確度計算工具

kf = KFold(n_splits=5,                                # 設定 K 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(train_x)                              # 給予資料範圍

test_pred_y = model.predict(test_x)

# 讀id
test_passenger_ids = df['PassengerId']

# 將預測結果轉為 DataFrame
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': test_pred_y
})

# 將 DataFrame 寫入 CSV 檔案
submission_df.to_csv('submission.csv', index=False)


# 請勿更動此區塊程式碼

EXECUTION_END_TIME = time.time() # 計算執行時間
print('total execution time: {}'.format(EXECUTION_END_TIME - EXECUTION_START_TIME))