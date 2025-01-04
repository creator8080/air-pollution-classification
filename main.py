
"""def win1(s):
    return s+1 >= 42 or s+2 >= 42 or s*2 >= 42

def lose1(s):
    return (not(win1(s))) and (win1(s+1) and win1(s+2) and win1(s*2))

def win2(s):
    return lose1(s+1) or lose1(s+2) or lose1(s*2)

def lose2(s):
    return win2(s+1) and win1(s+2) or win1(s+1) and win2(s+2) or win2(s*2) and win1(s+1) or win2(s+1) and win1(s*2) or win2(s*2) and win1(s+2) or win2(s+2) and win1(s*2)

for i in range(1,42):

    if lose1(i):
        print(i)

"""
"""
def f19(x,y,h):
    if (h ==2 or h == 4)  and x + y >= 41:
        return 1
    elif h == 4 and x+y < 41:
        return 0
    elif x+y >= 41 and h < 4:
        return 0
    else:
        if h % 2 == 1:
            return f19(x+1,y,h+1) or f19(x*2,y,h+1) or f19(x+2,y,h+1)
        else:
            return f19(x+1,y,h+1) and f19(x*2,y,h+1) and f19(x+2,y,h+1) and f19(x,y*2,h+1)

for x in range(1,32):
    if f19(x,8,0) == 1:
        print(x)

"""

"""from functools import lru_cache 
def moves(x): 
    return x+1, 2*x, x+2
@lru_cache (None) 
def game(x):
    if any (m>=42 for m in moves(x)): return "WIN1" 
    if all (game (m) == "WIN1" for m in moves(x)): return "LOSS1"
    if any (game(m) == "LOSS1" for m in moves(x)): return "WIN2"
    if all(game (m) == "WIN1" or game(m) == "WIN2" for m in moves(x)): return "LOSS12"
for x in range(1,41):
    if game(x) == "WIN2":
        print('Для задания 20 ответ', x)
    if game(x) == "LOSS12":
        print('Для задания 21 ответ', x)"""


"""import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor

df = pd.read_csv("data.csv")

scaler = MinMaxScaler()

param_grid = {
    'n_estimators':[200,300,400],
    'learning_rate': [0.005,0.01, 0.1], 
    'max_depth': [1,2,3], 
    'min_samples_split': [1,2, 5]
}

df['gender'] = df['gender'].replace({'female':1,'male':2})
df['race/ethnicity'],_ = pd.factorize(df['race/ethnicity'])
df['parental level of education'],_ = pd.factorize(df['parental level of education'])
df['lunch'],_ = pd.factorize(df['lunch'])
df['test preparation course'],_ = pd.factorize(df['test preparation course'])
df['math score'] = scaler.fit_transform(df[['math score']])


X = df[['gender','race/ethnicity','parental level of education','lunch','test preparation course']]
y = df['math score']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

model = GradientBoostingRegressor(random_state=42)

cv_model = GridSearchCV(model,param_grid,cv=3,scoring='neg_mean_squared_error')

cv_model.fit(X_train,y_train)

print("Best parameters:", cv_model.best_params_)
best_model = cv_model.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_pred,y_test)

print(f'MSE of this model is: {mse}')
print(f"MAE: {mae:.4f}")  
print(f"R²: {r2:.4f}")

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
baseline_mse = mean_squared_error(y_test, dummy.predict(X_test))
print(f"Baseline MSE: {baseline_mse:.4f}")
"""


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
import numpy as np

# Чтение данных из CSV файла
df = pd.read_csv('data.csv')

# Преобразование столбца 'Air Quality' в числовые категории
df['Air Quality'], _ = pd.factorize(df['Air Quality'])

# Определение диапазонов гиперпараметров для GridSearch
param_grid = {
    'n_estimators': [100],  # Количество деревьев в модели
    'max_depth': [3]  # Глубина каждого дерева
}

random_state = 42

# Определение признаков и целевой переменной
X = df[['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']]
y = df['Air Quality']

# Разделение данных на обучающую и тестовую выборки (25% данных для теста)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

# Инициализация модели (GradientBoosting)
model = GradientBoostingClassifier(random_state=random_state)

# Настройка GridSearchCV для поиска оптимальных гиперпараметров
cv_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')

# Обучение модели с использованием кросс-валидации
cv_search.fit(X_train, y_train)

# Получение лучших параметров и самой лучшей модели
best_params = cv_search.best_params_
best_model = cv_search.best_estimator_

# Прогнозирование на тестовых данных
y_pred = best_model.predict(X_test)

# Вывод лучших параметров модели
print(f'Best params: {best_params}')

# Оценка точности, точности по классу и F1-меры
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_pred, y_test, average='weighted')
f1 = f1_score(y_pred, y_test, average='weighted')

# Оценка матрицы ошибок и отчет по классификации
con_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Вывод метрик модели
print(f'Accuracy of this model: {accuracy}')
print(f'Precision of this model: {precision}')
print(f'F-1 score of this model: {f1}')
print(f'Confusion matrix of this model: \n {con_matrix}')
print(f'Classification report of this model: \n {class_report}')


# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(con_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Визуализация важности признаков
feature_importances = best_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Построение ROC кривых
y_probs = best_model.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 6))

for i in range(len(np.unique(y))):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Точность по классам
class_report = classification_report(y_test, y_pred, output_dict=True)

# Получаем точность для каждого класса
accuracy_per_class = {key: class_report[key]['precision'] for key in class_report if key.isdigit()}
class_names = [f'Class {i}' for i in accuracy_per_class.keys()]
accuracies = list(accuracy_per_class.values())

plt.figure(figsize=(8, 6))
sns.barplot(x=class_names, y=accuracies, palette='viridis')
plt.title('Accuracy per Class')
plt.xlabel('Classes')
plt.ylabel('Accuracy')
plt.show()
