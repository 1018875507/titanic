import pandas as pd
df = pd.read_csv(r"train.csv")
#删除PassengerId
df.drop('PassengerId', axis=1, inplace=True)

#填补缺失值

df['Fare'].fillna(df['Fare'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
#查看缺失值 查看数据字段信息
#print(df.isnull().sum())
#print(df.info())

#先对Embarked、Sex以及Pclass等用dummy处理
df = pd.get_dummies(df, columns=['Embarked', 'Sex', 'Pclass'])
#票价分级处理我们可以尝试将Fare分桶处理,使用qcut函数。qcut是根据这些值的频率来选择箱子的均匀间隔，每个箱子中含有的数的数量是相同的;
df['qcut_fare'], bins = pd.qcut(df['Fare'], 5, retbins=True)
df['qcut_fare_2fact'] = pd.factorize(df['qcut_fare'])[0]
tmp_fare_lv = pd.get_dummies(df['qcut_fare_2fact']).rename(columns=lambda x: 'Fare_lv_' + str(x))
df = pd.concat([df, tmp_fare_lv], axis=1)
df.drop(['qcut_fare', 'qcut_fare_2fact'], axis=1, inplace=True)

#名字处理：提取名字的称呼
#对名字Name进行处理，提取称呼，如Mr, Mrs, Miss等；

df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df['Title'] = df['Name'].apply(lambda x : x.split(',')[1].split('.')[0].strip())
titleDict = {
"Capt": "Officer",
"Col": "Officer",
"Major": "Officer",
"Jonkheer": "Royalty",
"Don": "Royalty",
"Sir": "Royalty",
"Dr": "Officer",
"Rev": "Officer",
"the Countess": "Royalty",
"Dona": "Royalty",
"Mme": "Mrs",
"Mlle": "Miss",
"Ms": "Mrs",
"Mr": "Mr",
"Mrs": "Mrs",
"Miss": "Miss",
"Master": "Master",
"Lady": "Royalty"
}
df['Title'] = df['Title'].map(titleDict)

#one_hot编码
df['Title'] = pd.factorize(df['Title'])[0]
title_dummies_df = pd.get_dummies(df['Title'], prefix=df[['Title']].columns[0])
df = pd.concat([df, title_dummies_df], axis=1)
#添加一列：提取名字长度
df['len_name'] = df['Name'].apply(len)

#Cabin处理
#Cabin缺失值过多，将其分为有无两类，进行编码，如果缺失，即为0，否则为1;
df.loc[df.Cabin.isnull(), 'Cabin'] = 'nan'
df['Cabin'] = df.Cabin.apply(lambda x: 0 if x == 'nan' else 1)

#Ticket处理
#Ticket有字母和数字之分，对于不同的字母，可能在很大程度上就意味着船舱等级或者不同船舱的位置，也会对Survived产生一定的影响，所以我们将Ticket中的字母分开，为数字的部分则分为一类。

df['Ticket_latter'] = df['Ticket'].apply(lambda x: x.split(' ')[0].strip())
df['Ticket_latter'] = df['Ticket_latter'].apply(lambda x: 'Latter' if x.isnumeric() == False else x)
df['Ticket_latter'] = pd.factorize(df['Ticket_latter'])[0]

#利用随机森林预测Age缺失值
#统一采用RandomForestRegressor(n_estimators=1000, n_jobs=-1)；
from sklearn.ensemble import RandomForestRegressor

missing_age = df.drop(['Survived', 'Name', 'Ticket'], axis=1) # 去除字符串类型的字段
missing_age_train = missing_age[missing_age['Age'].notnull()]
missing_age_test = missing_age[missing_age['Age'].isnull()]

X_train = missing_age_train.iloc[:, 1:]
y_train = missing_age_train.iloc[:, 0]
X_test = missing_age_test.iloc[:, 1:]

rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
rfr.fit(X_train, y_train)
y_predict = rfr.predict(X_test)
df.loc[df['Age'].isnull(), 'Age'] = y_predict
# 各特征与Survived的相关系数排序
df.corr()['Survived'].abs().sort_values(ascending=False)
df.to_csv(r"shuju.csv")
