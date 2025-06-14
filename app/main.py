import pickle
import joblib
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Request
from pydantic import BaseModel

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os

from geopy.geocoders import Nominatim
from geopy.distance import distance

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
import re
import math

with open("models/random_forest_model_06_05.pkl", "rb") as f:
    numeric_model = joblib.load(f)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))

AVG_PRICE_BY_DISTRICT = {
    0: 29729.294118,
    1: 61311.111111,
    2: 43573.643411,
    3: 28000,
    4: 28687.07483,
    5: 24375.2,
    6: 25852.937716,
    7: 32737.307692,
    8: 19670,
    9: 31224.722772,
    10: 54232.876712
}

# Словарь положительных слов с весами
POSITIVE_WORDS_WEIGHTED = {
    "большой": 0.02, "просторный": 0.02, "светлый": 0.02, "теплый": 0.02, "комфортный": 0.02, "тихий": 0.02,
    "отличный": 0.02, "красивый": 0.02, "хороший": 0.02, "ухоженный": 0.02, "ремонт": 0.02, "видеонаблюдение": 0.02, "изолированный": 0.02,
    "необходимый": 0.02, "солнечный": 0.02, "раздельный": 0.02, "качественный": 0.02,

    "новый": 0.05, "современный": 0.05, "охраняемый": 0.05, "новостройка": 0.05, "косметический": 0.05, "подземный": 0.05,
    "апартамент": 0.05, "жк": 0.05, "закрытый": 0.05, "двухуровневый": 0.05, "сплит": 0.05, "керамогранит": 0.05, "охрана": 0.05,
    "инфраструктура": 0.05, "жилой": 0.05, "консьерж": 0.05,

    "элитный": 0.08, "престижный": 0.08, "панорамный": 0.08, "евроремонт": 0.08, "дизайнер": 0.08, "дизайн": 0.08, "машиноместо": 0.08,
    "архитектура": 0.08, "интерьеры": 0.08, "роскошный": 0.08, "немецкий": 0.08, "гардеробный": 0.08, "гардероб": 0.08, "уборка": 0.08,
    "химчистка": 0.08, "бассейн": 0.08, "тренажерный": 0.08,

    "меблированный": 0.03, "оборудовать": 0.03, "укомплектовать": 0.03, "оснащать": 0.03, "полностью": 0.03, "комплектация": 0.03, "полный": 0.03,
    "балкон": 0.03, "мебель": 0.03, "высокий": 0.03, "техника": 0.03, "встроенный": 0.03, "кухонный": 0.03, "стиральный машина": 0.03,
    "посудомоечный": 0.02, "удобства": 0.03, "кондиционер": 0.03, "лифт": 0.03, "лобби": 0.03, "витражный": 0.03, "телевизор": 0.03, "лоджия": 0.03,

    "рядом": 0.02, "развитый": 0.02, "удобный": 0.02, "детский": 0.02, "шаговый": 0.02, "центр": 0.06, "метро": 0.02,
    "университет": 0.02, "школа": 0.02, "торговый": 0.02, "близко": 0.02, "пеший": 0.02, "доступность": 0.02, "магазин": 0.02,
    "площадка": 0.02, "вид": 0.02, "новат": 0.02, "сквер": 0.02, "трц": 0.02, "тц": 0.02,

    "собственник": 0.02, "включить": 0.05,
}

# Словарь отрицательных слов с весами
NEGATIVE_WORDS_WEIGHTED = {
    "старый": -0.05, "некрасивый": -0.05, "плохой": -0.05, "деревянный": -0.05, "шумный": -0.05, "шум": -0.05, "последний": -0.05,

    "тесный": -0.05, "маленький": -0.05, "студия": -0.05, "комната": -0.05,

    "требует": -0.06, "без": -0.06, "нет": -0.06, "дополнительно": -0.06, "частично": -0.06, "стройка": -0.06, "недоделка": -0.06, "только": -0.06,
    "пустой": -0.06, "продаваться": -0.06, "минимальный": -0.06, "расход": -0.06,

    "далеко": -0.06,

    "риелтор": -0.02, "залог": -0.02,

    "общежитие": -0.08, "времянка": -0.08, "малосемейка": -0.08,

    "вредный": -0.05, "животное": -0.05, "дети": -0.05, "ребенок":  -0.05, "свой": -0.05, "пожелание": -0.05
}

ALL_WEIGHTS = {**POSITIVE_WORDS_WEIGHTED, **NEGATIVE_WORDS_WEIGHTED}

STOP_WORDS = ["квартира", "сдам", "сдаю", "сдаётся", "сдается", "аренда", "арендовать", "и", "в", "на", "по", "с", "для", "от", "до", "за", "к", "что", "как", "это", "также", "или"]

# Определяем категории и ключевые слова
categories = {
    "ремонт": ["ремонт", "капитальный", "новый", "евроремонт", "косметический"],
    "мебель": ["мебель", "меблированный", "шкаф", "диван", "кухня", "стол"],
    "техника": ["техника", "холодильник", "стиральный", "стиралка", "кондиционер", "плита"],
    "район": ["район", "центр", "спальный", "тихий", "оживлённый"],
    "транспорт": ["транспорт", "метро", "остановка", "транспорт", "развязка"]
}

def check_missing_categories(text):
    """Определяем, какие категории отсутствуют"""
    processed_text = preprocess_text(text)
    missing_categories = []
    
    for category, words in categories.items():
        if not any(word in processed_text for word in words):
            missing_categories.append(category)
    
    return missing_categories

def suggest_expansion(text):
    """Предлагает расширить описание, если оно слишком короткое"""
    words = text.split()
    
    if len(words) < 10:  # Проверяем длину текста
        return "Ваше описание довольно короткое. Добавьте больше информации."

    return ""


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = re.sub('\n', ' ', text)
    text = text.strip()  # Убирает пробелы по краям
    text = re.sub(r'\s+', ' ', text)  # Заменяет множественные пробелы
    tokens = word_tokenize(text)
    morph = pymorphy3.MorphAnalyzer()
    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    processed_text = ' '.join(lemmas)
    processed_text = ' '.join([word for word in processed_text.split() if word not in STOP_WORDS])

    return processed_text

# Функция для расчета суммарного веса
def calculate_weight(text):
    if not text:
        return 0
    words = text.split()
    return sum(ALL_WEIGHTS.get(word, 0) for word in words)

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    floor: int = Form(...),
    floor_count: int = Form(...),
    rooms_count: int = Form(...),
    total_meters: float = Form(...),
    district_encoded: int = Form(...),
    street: str = Form(...),
    building: int = Form(...),
    description: str = Form(...),
    description_toggle: str = Form("false"),
):
    description_toggle = description_toggle == "true"
    longitude, latitude = get_coordinates(street, building)
    distance_to_metro = get_distance_to_subway(longitude, latitude, get_nearest_subway(longitude, latitude))
    distance_to_center = get_distance_to_center(longitude, latitude)
    avg_price_by_district = AVG_PRICE_BY_DISTRICT[district_encoded]

    expansion = suggest_expansion(description)
    missing_list = check_missing_categories(description)
    missing = ""
    if missing_list:
        missing = "В объявлении отсутствуют данные о: " + ', '.join(missing_list) + ". Рекомендуется их добавить!"

    numerical_features = pd.DataFrame([[
        floor, floor_count, rooms_count, total_meters, latitude,
        longitude, distance_to_metro, avg_price_by_district, district_encoded, distance_to_center
    ]], columns=[
        'floor', 'floors_count', 'rooms_count', 'total_meters',
        'latitude', 'longitude', 'distance_to_metro', 'avg_price_by_district', 'district_encoded', 'distance_to_center'
    ])
    numerical_prediction = math.e ** numeric_model.predict(numerical_features)[0]
    if isinstance(numerical_prediction, (np.floating, np.integer)):
        numerical_prediction = float(numerical_prediction)
    prediction_with_text = numerical_prediction + numerical_prediction * calculate_weight(preprocess_text(description))

    return templates.TemplateResponse("index.html", {
        "request": request,
        "floor": floor,
        "floor_count": floor_count,
        "rooms_count": rooms_count,
        "total_meters": total_meters,
        "district_encoded": district_encoded,
        "street": street,
        "building": building,
        # и всё остальное:
        "numerical_prediction": numerical_prediction,
        "distance_to_metro": distance_to_metro,
        "distance_to_center": distance_to_center,
        "nearest_subway": get_nearest_subway(longitude, latitude),
        "longitude": longitude,
        "latitude": latitude,
        "prediction_with_text": prediction_with_text,
        "description": description,
        "description_toggle": str(description_toggle).lower(),
        "expansion": expansion,
        "missing": missing,
    })

def get_coordinates(street, building):
    geolocator = Nominatim(user_agent="my_app", timeout=10)
    location = geolocator.geocode(f"Новосибирск, {street} {building}")
    if location:
        return location.longitude, location.latitude
    else:
        # Handle the case where the location is not found
        # You can return a default value or raise an exception
        return None, None
    
def get_nearest_subway(longitude, latitude):
    coordinates_map = {
        "Площадь Ленина": (55.029621, 82.919064),
        "Гагаринская": (55.051278, 82.913944),
        "Студенческая": (54.989094, 82.905006),
        "Площадь Гарина-Михайловского": (55.035318, 82.898376),
        "Красный проспект": (55.041539, 82.916441),
        "Золотая Нива": (55.037718, 82.977199),
        "Березовая роща": (55.043933, 82.953488),
        "Площадь Маркса": (54.982724, 82.893274),
        "Речной вокзал": (55.008354, 82.937339),
        "Октябрьская": (55.019356, 82.939734),
        "Сибирская": (55.042860, 82.920070),
        "Заельцовская": (55.059438,82.912503),
        "Маршала Покрышкина": (55.043386, 82.935683),
    }
    nearest_subway = None
    min_distance = float('inf')
    for subway, coords in coordinates_map.items():
        dist = distance((latitude, longitude), coords).m
        if dist < min_distance:
            min_distance = dist
            nearest_subway = subway
    return nearest_subway

def get_distance_to_subway(longitude, latitude, subway):
    coordinates_map = {
        "Площадь Ленина": (55.029621, 82.919064),
        "Гагаринская": (55.051278, 82.913944),
        "Студенческая": (54.989094, 82.905006),
        "Площадь Гарина-Михайловского": (55.035318, 82.898376),
        "Красный проспект": (55.041539, 82.916441),
        "Золотая Нива": (55.037718, 82.977199),
        "Березовая роща": (55.043933, 82.953488),
        "Площадь Маркса": (54.982724, 82.893274),
        "Речной вокзал": (55.008354, 82.937339),
        "Октябрьская": (55.019356, 82.939734),
        "Сибирская": (55.042860, 82.920070),
        "Заельцовская": (55.059438,82.912503),
        "Маршала Покрышкина": (55.043386, 82.935683),
    }
    subway_coordinates = coordinates_map.get(subway)
    return distance((latitude, longitude), subway_coordinates).m

def get_distance_to_center(longitude, latitude):
    return distance((latitude, longitude), (55.036705, 82.929179)).m
