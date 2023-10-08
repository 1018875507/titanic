from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report

from sklearn import svm
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, classification_report
df = pd.read_csv(r"shuju.csv")
#删除PassengerId
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)

y = df.iloc[:,1:2]
X = df.iloc[:,2:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化模型
models = [
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    SVC(),
    MLPClassifier(),
    xgb.XGBClassifier(),
    RandomForestClassifier(n_estimators=11,random_state=42),
]

# 训练并测试每个模型
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='macro')
    print(f'Model: {model.__class__.__name__}, Macro F1 Score: {score}')
    print(classification_report(y_test, y_pred))




