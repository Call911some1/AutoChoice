import pandas as pd
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Функция для обработки аспектов
def process_aspects(data):
    positive_aspects = data['positive_aspects'].dropna().str.cat(sep=', ')
    negative_aspects = data['negative_aspects'].dropna().str.cat(sep=', ')
    
    positive_list = [aspect.strip() for aspect in positive_aspects.split(',') if aspect.strip()]
    negative_list = [aspect.strip() for aspect in negative_aspects.split(',') if aspect.strip()]
    
    aspect_counts = defaultdict(lambda: {'positive': 0, 'negative': 0})
    
    for aspect in positive_list:
        aspect_counts[aspect]['positive'] += 1
    
    for aspect in negative_list:
        aspect_counts[aspect]['negative'] += 1
    
    return aspect_counts

def create_wordcloud(frequency_dict, title, color_map='viridis', max_words=50, background_color='black'):
    if not frequency_dict:
        return None
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,  # Тёмный фон
        colormap=color_map,
        contour_color='white',  # Белая окантовка текста
        contour_width=1,
        max_words=max_words
    ).generate_from_frequencies(frequency_dict)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=20, color='white')  # Белый заголовок
    ax.axis('off')
    plt.tight_layout()
    return fig

def create_positive_wordcloud(aspect_counts, title):
    freq_dict = {aspect: counts['positive'] - counts['negative']
                for aspect, counts in aspect_counts.items()
                if counts['positive'] > counts['negative']}
    return create_wordcloud(freq_dict, title, color_map='coolwarm')  # Градиент для позитивных слов

def create_negative_wordcloud(aspect_counts, title):
    freq_dict = {aspect: counts['negative'] - counts['positive']
                for aspect, counts in aspect_counts.items()
                if counts['negative'] > counts['positive']}
    return create_wordcloud(freq_dict, title, color_map='Reds')  # Градиент для негативных слов

# def create_wordcloud(frequency_dict, title, color_map='viridis', max_words=50):
#     if not frequency_dict:
#         return None
#     wordcloud = WordCloud(
#         width=800,
#         height=400,
#         background_color='white',
#         colormap=color_map,
#         max_words=max_words
#     ).generate_from_frequencies(frequency_dict)
#     fig, ax = plt.subplots(figsize=(10, 7.5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.set_title(title, fontsize=20)
#     ax.axis('off')
#     plt.tight_layout()
#     return fig

# # Функции для создания позитивного и негативного облака слов
# def create_positive_wordcloud(aspect_counts, title):
#     freq_dict = {aspect: counts['positive'] - counts['negative']
#                 for aspect, counts in aspect_counts.items()
#                 if counts['positive'] > counts['negative']}
#     return create_wordcloud(freq_dict, title)

# def create_negative_wordcloud(aspect_counts, title):
#     freq_dict = {aspect: counts['negative'] - counts['positive']
#                 for aspect, counts in aspect_counts.items()
#                 if counts['negative'] > counts['positive']}
#     return create_wordcloud(freq_dict, title)

# Функция для отображения распределения оценок
def show_rating_distribution(data, brand, model):
    pass
    # fig, ax = plt.subplots()
    # sns.histplot(data['average_rating'], kde=True, bins=10, ax=ax)
    # ax.set_title(f'Распределение оценок для {brand.title()} {model.title()}')
    # ax.set_xlabel('Оценка')
    # ax.set_ylabel('Количество отзывов')
    # plt.tight_layout()
    # return fig

# Функция для отображения информации об автомобиле
def show_car_info(brand, model, years=None, df_reviews=None):
    if df_reviews is None:
        return "Данные отзывов не предоставлены.", None, None, None

    brand = brand.lower()
    model = model.lower()

    # Фильтрация данных по марке и модели
    data = df_reviews[(df_reviews['brand'] == brand) & (df_reviews['model'] == model)]

    if data.empty:
        message = f"Отзывов нет для {brand.title()} {model.title()}."
        return message, None, None, None

    if years:
        # Фильтрация по диапазону лет
        start_year, end_year = years
        filtered_data = data[(data['year'] >= start_year) & (data['year'] <= end_year)]

        if filtered_data.empty:
            # Если данных нет в диапазоне, берем ближайший год
            nearest_year_idx = (data['year'] - start_year).abs().idxmin()
            nearest_year_value = data.loc[nearest_year_idx, 'year']
            message = f"Нет данных в диапазоне {start_year}-{end_year}. Используем данные за ближайший год: {int(nearest_year_value)}."
            data = data[data['year'] == nearest_year_value]
        else:
            data = filtered_data
            message = None
    else:
        message = None

    # Расчет медианных значений
    average_rating = data['average_rating'].median()
    exterior_rating = data['exterior'].median()
    interior_rating = data['interior'].median()
    engine_rating = data['engine_rating'].median()
    driving_quality_rating = data['driving_quality'].median()

    # Создание списков параметров и значений, исключая NaN
    parameters = []
    values = []

    if not pd.isna(average_rating):
        parameters.append('Средний рейтинг')
        values.append(average_rating)
    if not pd.isna(exterior_rating):
        parameters.append('Оценка внешнего вида')
        values.append(exterior_rating)
    if not pd.isna(interior_rating):
        parameters.append('Оценка интерьера')
        values.append(interior_rating)
    if not pd.isna(engine_rating):
        parameters.append('Оценка двигателя')
        values.append(engine_rating)
    if not pd.isna(driving_quality_rating):
        parameters.append('Оценка качества вождения')
        values.append(driving_quality_rating)

    if not parameters:
        message = f"Нет данных оценок для {brand.title()} {model.title()}."
        return message, None, None, None

    # Создание таблицы с оценками
    rating_df = pd.DataFrame({'Параметр': parameters, 'Значение': values})

    # Округление и преобразование значений к целым числам
    rating_df['Значение'] = rating_df['Значение'].round().astype(int)

    # Обработка аспектов
    aspect_counts = process_aspects(data)

    pos_wordcloud_fig = create_positive_wordcloud(aspect_counts, f'Хвалят в {model.title()}')
    neg_wordcloud_fig = create_negative_wordcloud(aspect_counts, f'На что рекомендую обратить внимание')

    return message, rating_df, pos_wordcloud_fig, neg_wordcloud_fig
