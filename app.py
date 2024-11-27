import streamlit as st
import pandas as pd
import numpy as np
import joblib
import word_cloud_utils as wc_utils
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.ticker import FuncFormatter
matplotlib.use('Agg')
import mplcyberpunk

# Установка стиля Cyberpunk
plt.style.use("cyberpunk")

# Загрузка данных отзывов
@st.cache_data
def load_reviews_data(file_path='final_reviews_cleaned.csv'):
    df_reviews = pd.read_csv(file_path)
    df_reviews['brand'] = df_reviews['brand'].str.lower()
    df_reviews['model'] = df_reviews['model'].str.lower()
    return df_reviews

# Загрузка данных автомобилей
@st.cache_data
def load_data(file_path='Cars.csv'):
    df = pd.read_csv(file_path)
    df = process_car_data_1(df)
    df = preprocess_data(df)
    return df

# Загрузка модели
@st.cache_resource
def load_pipeline(pipeline_path='model_pipeline.pkl'):
    pipeline = joblib.load(pipeline_path)
    return pipeline

# Предобработка данных от Cars.csv до cars_filtered.csv
def process_car_data_1(df):
    # Фильтруем строки, где заголовок начинается с 'Продажа'
    df = df[df['title'].str.startswith('Продажа')]

    # Извлечение модели, года и местоположения из заголовка
    pattern = r'Продажа (.*?), (\d{4}) год в (.*?) -'
    df[['model', 'year', 'location']] = df['title'].str.extract(pattern)
    df.drop(columns=['title'], inplace=True)

    # Извлечение типа и объёма двигателя
    pattern = r'(\w+), (\d+\.\d+ л)'
    df[['type_engine', 'volume_engine']] = df['engine'].str.extract(pattern)

    # Удаление ненужных столбцов
    df.drop(columns=['engine', 'description'], inplace=True)

    # Обработка столбца 'power'
    df['power'] = df['power'].str.split(',').str[0]
    df['location'] = 'в ' + df['location']
    df['year'] = pd.to_numeric(df['year'])  # Год приводим к числовому значению

    # Упорядочивание столбцов
    new_order = [
        'model', 'year', 'price', 'location', 'type_engine', 'volume_engine',
        'power', 'transmission', 'drive', 'body_type', 'color', 'mileage',
        'steering', 'generation', 'complectation', 'additional_info', 'image_url', 'listing_url'
    ]

    # Структура по новому порядку и удаление пустых значений в столбце 'model'
    df = df[new_order]
    df = df[df['model'].notna() & (df['model'] != '')]
    df.reset_index(drop=True, inplace=True)

    return df

# Предобработка данных
def preprocess_data(df):
    # Проверяем, есть ли колонка 'model'
    if 'model' in df.columns:
        df.rename(columns={'model': 'full_model'}, inplace=True)
        df[['brand', 'model']] = df['full_model'].str.split(n=1, expand=True)
        df.drop(columns=['full_model'], inplace=True)
    else:
        st.error("Колонка 'model' отсутствует в данных.")
        return df

    # Остальная обработка
    df['price'] = df['price'].str.replace('\xa0', '', regex=False).str.replace('₽', '', regex=False).str.replace(' ', '', regex=False).astype(int)
    df['volume_engine'] = df['volume_engine'].str.replace('\xa0', '', regex=False).str.replace(' л', '', regex=False).astype(float)
    df['volume_engine'] = df['volume_engine'].apply(lambda x: x if x <= 7 else 7)
    df['power'] = df['power'].replace('Нет данных', np.nan)
    df = df.dropna(subset=['power'])
    df['power'] = df['power'].str.replace('\xa0', '', regex=False).str.replace('л.с.', '', regex=False).astype(float)
    df = df[(df['power'] >= 40) & (df['power'] <= 600)]
    df['body_type'] = df['body_type'].replace({
        'хэтчбек 5 дв.': 'хэтчбек',
        'хэтчбек 3 дв.': 'хэтчбек',
        'джип/suv 5 дв.': 'джип/SUV',
        'джип/suv 3 дв.': 'джип/SUV',
        'джип/suv': 'джип/SUV',
        'джип/SUV': 'джип/SUV',
    })
    df = df[df['mileage'] != 'Нет данных']
    df['mileage'] = df['mileage'].str.strip().str.lower()
    df = df[~df['mileage'].str.contains('безпробегапорф')]
    df['mileage'] = df['mileage'].replace({'новый автомобиль': '0'})
    df['mileage'] = df['mileage'].str.replace(r'\s+', '', regex=True).str.extract(r'(\d+)').astype(float)
    df = df.dropna(subset=['mileage'])
    df['transmission'] = df['transmission'].replace('АКПП', 'автомат').fillna('Нет данных')
    df = df[df['transmission'] != 'Нет данных']
    df['drive'] = df['drive'].fillna('передний')
    df['type_engine'] = df['type_engine'].fillna('бензин')
    df['color'] = df['color'].replace({'Нет данных': 'черный', 'чёрный': 'черный'})
    df = df[df['steering'] != 'Нет данных']
    df.reset_index(drop=True, inplace=True)
    return df

# Создаём функции форматирования осей
def price_formatter(x, pos):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f} млн'
    elif x >= 1000:
        return f'{x/1000:.0f} тыс.'
    else:
        return f'{x:.0f}'

def mileage_formatter(x, pos):
    if x >= 100_000:
        return f'{x/1000:.0f} тыс.'
    elif x >= 1000:
        return f'{x/1000:.0f} тыс.'
    else:
        return f'{x:.0f}'

# Добавление графиков с визуализацией данных (убрали фильтрацию по году)
def plot_price_distribution(df, brand, model):
    df_filtered = df[(df['brand'] == brand) & (df['model'] == model)]
    if df_filtered.empty or len(df_filtered) < 5:
        st.write("Недостаточно данных для построения графика распределения цен.")
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(df_filtered['price'], kde=True, bins=20)
    plt.title(f'Распределение цен для {brand} {model}')
    plt.xlabel('Цена (руб.)')
    plt.ylabel('Количество объявлений')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(price_formatter))
    plt.tight_layout()
    # Добавление Cyberpunk стиля
    mplcyberpunk.add_glow_effects()
    st.pyplot(plt.gcf())
    plt.close()

def plot_price_vs_mileage(df, brand, model):
    df_filtered = df[(df['brand'] == brand) & (df['model'] == model)]
    if df_filtered.empty or len(df_filtered) < 5:
        st.write("Недостаточно данных для построения графика зависимости цены от пробега.")
        return
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=df_filtered['mileage'], 
        y=df_filtered['price'], 
        c=df_filtered['year'],  # Используем год выпуска для цвета
        cmap='viridis',         # Палитра для цветового отображения
        alpha=0.8,              # Прозрачность точек
        edgecolor='w',          # Белая окантовка для точек
        s=80                    # Размер точек
    )
    
    plt.colorbar(scatter, label='Год выпуска')  # Добавляем легенду для цветовой шкалы
    plt.title(f'Зависимость цены от пробега для {brand} {model}')
    plt.xlabel('Пробег (км)')
    plt.ylabel('Цена (руб.)')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(mileage_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(price_formatter))
    plt.tight_layout()

    # Добавляем Cyberpunk стиль
    mplcyberpunk.add_glow_effects()

    st.pyplot(plt.gcf())
    plt.close()


# def plot_price_vs_mileage(df, brand, model):
#     df_filtered = df[(df['brand'] == brand) & (df['model'] == model)]
#     if df_filtered.empty or len(df_filtered) < 5:
#         st.write("Недостаточно данных для построения графика зависимости цены от пробега.")
#         return
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='mileage', y='price', data=df_filtered)
#     plt.title(f'Зависимость цены от пробега для {brand} {model}')
#     plt.xlabel('Пробег (км)')
#     plt.ylabel('Цена (руб.)')
#     plt.gca().xaxis.set_major_formatter(FuncFormatter(mileage_formatter))
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(price_formatter))
#     plt.tight_layout()
#     # Добавление Cyberpunk стиля
#     mplcyberpunk.add_glow_effects()
#     st.pyplot(plt.gcf())
#     plt.close()

# Функция для поиска похожих автомобилей с пошаговым расширением критериев
# (оставляем без изменений)
def find_similar_cars(user_input, df_original):
    predicted_price = user_input.get('predicted_price', None)
    if predicted_price is None:
        st.error("Предсказанная цена отсутствует.")
        return pd.DataFrame()

    user_brand = user_input['brand'].lower().strip()
    user_model = user_input['model'].lower().strip()
    user_year = user_input['year']
    user_mileage = user_input['mileage']

    # Список критериев с постепенным ослаблением
    criteria_list = [
        {
            'year_diff': 0,
            'mileage_diff_pct': 0.10,
            'price_diff_pct': 0.10,
            'include_older_years': False
        },
        {
            'year_diff': 1,
            'mileage_diff_pct': 0.15,
            'price_diff_pct': 0.15,
            'include_older_years': False
        },
        {
            'year_diff': 2,
            'mileage_diff_pct': 0.20,
            'price_diff_pct': 0.20,
            'include_older_years': False
        },
        {
            'year_diff': 2,
            'mileage_diff_pct': 0.20,
            'price_diff_pct': 0.20,
            'include_older_years': True
        },
        {
            'year_diff': 5,
            'mileage_diff_pct': 0.30,
            'price_diff_pct': 0.30,
            'include_older_years': True
        },
    ]

    for criteria in criteria_list:
        year_diff = criteria['year_diff']
        mileage_diff_pct = criteria['mileage_diff_pct']
        price_diff_pct = criteria['price_diff_pct']
        include_older_years = criteria['include_older_years']

        mileage_diff = user_mileage * mileage_diff_pct
        price_diff = predicted_price * price_diff_pct

        # Условие по году выпуска
        if include_older_years:
            year_condition = (
                (df_original['year'] >= user_year - year_diff) &
                (df_original['year'] <= user_year + year_diff)
            )
        else:
            year_condition = (
                (df_original['year'] >= user_year) &
                (df_original['year'] <= user_year + year_diff)
            )

        filtered_cars = df_original[
            (df_original['brand'].str.lower() == user_brand) &
            (df_original['model'].str.lower() == user_model) &
            year_condition &
            (abs(df_original['mileage'] - user_mileage) <= mileage_diff) &
            (abs(df_original['price'] - predicted_price) <= price_diff)
        ]

        if len(filtered_cars) >= 5:
            return filtered_cars.sort_values(by='price').head(5)

    # Если недостаточно объявлений, возвращаем наиболее близкие по цене автомобили
    filtered_cars = df_original[
        (df_original['brand'].str.lower() == user_brand) &
        (df_original['model'].str.lower() == user_model)
    ].copy()

    if filtered_cars.empty:
        return pd.DataFrame()

    filtered_cars['price_diff'] = abs(filtered_cars['price'] - predicted_price)
    return filtered_cars.sort_values(by='price_diff').head(5)

# Основная функция
def main():
    st.title("Оценка стоимости автомобиля")

    # Загрузка данных и модели
    df = load_data()
    pipeline = load_pipeline()
    df_reviews = load_reviews_data()

    # Проверка на наличие необходимых колонок
    if 'brand' not in df.columns:
        st.error("Колонка 'brand' отсутствует в данных.")
        st.stop()

    # Создание двух колонок
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Марка", options=sorted(df['brand'].unique()))
        df_brand = df[df['brand'] == brand]

        model_input = st.selectbox("Модель", options=sorted(df_brand['model'].unique()))
        df_model = df_brand[df_brand['model'] == model_input]

        year = st.number_input("Год выпуска", min_value=int(df_model['year'].min()), max_value=int(df_model['year'].max()), step=1)
        mileage = st.number_input("Пробег (км)", min_value=0, max_value=int(df_model['mileage'].max()), step=1000)

        # Объёмы двигателя
        available_volumes = sorted(df_model['volume_engine'].round(1).unique())
        volume_engine = st.selectbox("Объем двигателя (л)", options=available_volumes)

        # Мощность двигателя
        available_powers = sorted(df_model['power'].astype(int).unique())
        power = st.selectbox("Мощность двигателя (л.с.)", options=available_powers)

    with col2:
        transmission_options = sorted(df_model['transmission'].unique())
        transmission = st.selectbox("Коробка передач", options=transmission_options)

        drive_options = sorted(df_model['drive'].unique())
        drive = st.selectbox("Привод", options=drive_options)

        # Если количество уникальных значений `body_type` меньше 2, используем все доступные опции
        if len(df_model['body_type'].unique()) < 2:
            body_type_options = sorted(df['body_type'].unique())
        else:
            body_type_options = sorted(df_model['body_type'].unique())
        body_type = st.selectbox("Тип кузова", options=body_type_options)

        # Аналогично для цвета
        if len(df_model['color'].unique()) < 2:
            color_options = sorted(df['color'].unique())
        else:
            color_options = sorted(df_model['color'].unique())
        color = st.selectbox("Цвет", options=color_options)

        type_engine_options = sorted(df_model['type_engine'].unique())
        type_engine = st.selectbox("Тип двигателя", options=type_engine_options)

        steering_options = sorted(df_model['steering'].unique())
        steering = st.selectbox("Руль", options=steering_options)

        location = st.text_input("Локация", value="в Москве")

    if st.button("Рассчитать стоимость"):
        # Собираем входные данные
        input_data = {
            'brand': brand,
            'model': model_input,
            'year': year,
            'mileage': mileage,
            'volume_engine': volume_engine,
            'power': power,
            'transmission': transmission,
            'drive': drive,
            'body_type': body_type,
            'color': color,
            'type_engine': type_engine,
            'steering': steering,
            'location': location
        }

        # Прогнозирование цены
        try:
            predicted_price = pipeline.predict(pd.DataFrame([input_data]))[0]
            rounded_price = int(round(predicted_price / 10000) * 10000)
            st.markdown(
                f"<div style='padding: 10px; background-color: #dff0d8; border-radius: 5px;'>"
                f"<h3 style='text-align: center; color: #3c763d;'>Предполагаемая стоимость автомобиля:</h3>"
                f"<h2 style='text-align: center; color: #3c763d;'>{rounded_price:,} руб.</h2></div>",
                unsafe_allow_html=True
            )
            input_data['predicted_price'] = rounded_price  # Добавляем округленную цену в данные пользователя
        except Exception as e:
            st.error(f"Ошибка при прогнозировании цены: {e}")
            return

        # Поиск похожих автомобилей
        similar_cars = find_similar_cars(input_data, df)
        st.write("Похожие объявления:")
        if not similar_cars.empty:
            for idx, (_, row) in enumerate(similar_cars.iterrows(), 1):
                st.write(f"{idx}. {row['brand']} {row['model']} {int(row['year'])}, {int(row['mileage']):,} км — {row['price']:,} руб.")
                if pd.notna(row['image_url']):
                    try:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.image(row['image_url'], width=300)
                    except:
                            st.write("Изображение недоступно.")
                else:
                    st.write("Изображение отсутствует.")
                st.markdown(f"[Ссылка на объявление]({row['listing_url']})")
                st.write("---")
        else:
            st.write("Не найдено похожих объявлений.")

        # Отображение графиков
        st.subheader("Дополнительная информация")

        plot_price_distribution(df, brand, model_input)
        plot_price_vs_mileage(df, brand, model_input)

        # Отображение информации об автомобиле и облаков слов
        message, rating_df, pos_wordcloud_fig, neg_wordcloud_fig = wc_utils.show_car_info(
            brand, model_input, df_reviews=df_reviews
        )

        if message:
            st.warning(message)

        if rating_df is not None:
            st.write(f"Средние оценки для {brand.title()} {model_input.title()}:")
            st.table(rating_df)

        if pos_wordcloud_fig:
            st.pyplot(pos_wordcloud_fig)
        else:
            st.write("Недостаточно информации по отзывам, чтобы выделить, что хвалят в автомобиле.")

        if neg_wordcloud_fig:
            st.pyplot(neg_wordcloud_fig)
        else:
            st.write("Недостаточно информации по отзывам, чтобы выделить, на что рекомендую обратить внимание в автомобиле.")

if __name__ == "__main__":
    main()
