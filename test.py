import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer

# 讀取資料
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 將 train 和 test 資料集合併，方便資料預處理
data_df = pd.concat([train_df, test_df], ignore_index=True)

# 將 'Name', 'Ticket', 'Cabin' 這三個特徵刪除，因為這些特徵對於生存預測沒有幫助
data_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 用中位數填補 'Age' 特徵的缺失值
imputer = KNNImputer(n_neighbors=5)
data_df['Age'] = imputer.fit_transform(data_df[['Age']])
data_df['Fare'] = imputer.fit_transform(data_df[['Fare']])

# 用最常出現的值填補 'Embarked' 特徵的缺失值
most_frequent = data_df['Embarked'].mode()[0]
data_df['Embarked'].fillna(most_frequent, inplace=True)

# 將 'Sex' 特徵用 LabelEncoder 轉換成數值特徵
label_encoder = LabelEncoder()
data_df['Sex'] = label_encoder.fit_transform(data_df['Sex'])

# 將 'Embarked' 特徵用 OneHotEncoder 轉換成數值特徵
onehot_encoder = OneHotEncoder()
embarked_onehot = onehot_encoder.fit_transform(data_df[['Embarked']]).toarray()
data_df[['Embarked_C', 'Embarked_Q', 'Embarked_S']] = embarked_onehot

# 將 'Pclass' 特徵用 OneHotEncoder 轉換成數值特徵
pclass_onehot = onehot_encoder.fit_transform(data_df[['Pclass']]).toarray()
data_df[['Pclass_1', 'Pclass_2', 'Pclass_3']] = pclass_onehot

# 將資料集切成訓練集和測試集
train_data = data_df.iloc[:len(train_df)]
test_data = data_df.iloc[len(train_df):]

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

# 定義要使用的特徵欄位
features = ['Sex', 'Age', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'SibSp', 'Parch', 'FamilySize']

# # 建立模型
# decision_tree = DecisionTreeClassifier(max_depth=5, random_state=4002)

# 分割資料集，使用 K-fold cross validation 評估模型

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(decision_tree, train_data[features], train_df['Survived'], cv=kf)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=5100, max_depth=5, random_state=42)
scores = cross_val_score(random_forest, train_data[features], train_df['Survived'], cv=kf)

print('Cross-validation scores:', scores)
print('Mean:', scores.mean())
print('Standard deviation:', scores.std())

# 訓練模型
# decision_tree.fit(train_data[features], train_df['Survived'])
random_forest.fit(train_data[features], train_df['Survived'])

# 在測試集上進行預測
# predictions = decision_tree.predict(test_data[features])
predictions = random_forest.predict(test_data[features])

# 將預測結果轉換成 DataFrame 格式
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})

# 將預測結果儲存成 csv 檔案
submission_df.to_csv('submission.csv', index=False)

