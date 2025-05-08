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
    "просторный": 0.05, "светлый": 0.04, "новый": 0.05, "современный": 0.05, "уютный": 0.02,
    "теплый": 0.02, "элитный": 0.05, "ремонт": 0.04, "панорамный": 0.03, "балкон": 0.01,
    "мебель": 0.03, "тихий": 0.015, "охраняемый": 0.04, "рядом": 0.02,
    "развитый": 0.03, "удобный": 0.01, "высокий": 0.02, "техника": 0.03, "кухонный": 0.03, "детский": 0.02,
    "стиральный машина": 0.03, "шаговый": 0.02, "комфортный": 0.02,
    "косметический ремонт": 0.05, "центр": 0.06, "метро": 0.04, "собственник": 0.015,
    "новостройка": 0.04, "евроремонт": 0.05, "дизайнер": 0.08, "дизайн": 0.08,
    "меблированный": 0.03, "удобства": 0.02, "комплектация": 0.03,
    "подземный": 0.03, "жк": 0.015, "видеонаблюдение": 0.025, "кондиционер": 0.02,
    "закрытый": 0.025, "посудомоечный": 0.02, "престижный": 0.035, "машиноместо": 0.08, "университет": 0.03, "школа": 0.03, "торговый": 0.03, 
    "подземный": 0.08, "апартамент": 0.05, "близко": 0.03, "отличный": 0.02, "красивый": 0.02, "хороший": 0.02, "пеший": 0.02, "доступность": 0.02
}

# Словарь отрицательных слов с весами
NEGATIVE_WORDS_WEIGHTED = {
    "старый": -0.035, "требует": -0.06, "некрасивый": -0.03,
    "шумный": -0.025, "далеко": -0.03, "плохой": -0.05,
    "маленький": -0.035, "без": -0.035, "последний": -0.035,
    "стройка": -0.04, "тесный": -0.04,  "дополнительно": -0.03,
    "риелтор": -0.03, "депозит": -0.03, "шум": -0.035, "залог": -0.025,
    "общежитие": -0.045, "деревянный": -0.035,
    "времянка": -0.045, "вредный": -0.03, "животное": -0.025, "дети": -0.015, "ребенок":  -0.015, "свой": -0.08
}

ALL_WEIGHTS = {**POSITIVE_WORDS_WEIGHTED, **NEGATIVE_WORDS_WEIGHTED}

STOP_WORDS = ["квартира", "сдам", "сдаю", "сдаётся", "сдается", "аренда", "арендовать", "и", "в", "на", "по", "с", "для", "от", "до", "за", "к", "что", "как", "это", "также", "или"]

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
        "description_toggle": str(description_toggle).lower()
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
