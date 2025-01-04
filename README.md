# air-pollution-classification
**Air Quality and Pollution Assessment**

**Описание**

Этот проект использует данные о загрязнении воздуха и экологические метрики для предсказания качества воздуха с помощью алгоритма градиентного бустинга (GradientBoostingClassifier). Модель предсказывает класс качества воздуха на основе нескольких факторов, таких как температура, влажность, концентрации загрязняющих веществ и другие.
Данные

**Данные содержат следующие столбцы:**

    Temperature: Температура (°C)
    Humidity: Влажность
    PM2.5: Частицы PM2.5
    PM10: Частицы PM10
    NO2: Оксид азота
    SO2: Диоксид серы
    CO: Угарный газ
    Proximity_to_Industrial_Areas: Расстояние до промышленных зон
    Population_Density: Плотность населения
    Air Quality: Качество воздуха (целевой столбец)

**Как запустить**

    Клонируй репозиторий:

git clone https://github.com/username/repository-name.git

Установи зависимости:

pip install -r requirements.txt

Скачай данные и размести их в папке с проектом (например, data.csv).

Запусти модель:

    python model.py

**Оценка модели**

Модель оценивается по:

    Accuracy: точность предсказаний.
    Precision: точность по классам.
    F1-Score: среднее гармоническое точности и полноты.
    Confusion Matrix: матрица ошибок
    Classification Report: отчет по классификации

Требования

    pandas
    scikit-learn
    matplotlib

Автор

Проект разработан creator8080.
