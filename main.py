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
    'n_estimators': [10,50,100],  # Количество деревьев в модели
    'max_depth': [3,4,5]  # Глубина каждого дерева
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
