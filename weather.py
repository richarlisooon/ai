import logging
import requests
from datetime import datetime


class WeatherService:
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    def __init__(self, api_key: str, logger: logging.Logger):
        self.api_key = api_key
        self.logger = logger

    def get_current(self, city: str) -> str:
        url = f"{self.BASE_URL}/weather?q={city}&appid={self.api_key}&units=metric&lang=ru"
        try:
            response = requests.get(url, timeout=15, headers=self.HEADERS)
            if response.status_code == 200:
                data = response.json()
                return (
                    f"🌆 Город: {data['name']}\n"
                    f"⛅ Погода: {data['weather'][0]['description'].capitalize()}\n"
                    f"🌡 Температура: {data['main']['temp']:.1f}°C (ощущается как {data['main']['feels_like']:.1f}°C)\n"
                    f"💧 Влажность: {data['main']['humidity']}%\n"
                    f"🌬 Ветер: {data['wind']['speed']:.1f} м/с"
                )
            elif response.status_code == 404:
                return f"❌ Не могу найти город {city}"
            else:
                return "❌ Ошибка получения погоды"
        except requests.exceptions.Timeout:
            return "❌ Превышено время ожидания ответа от сервиса погоды"
        except requests.exceptions.ConnectionError:
            return "❌ Не удалось подключиться к сервису погоды"

    def get_forecast(self, city: str) -> str:
        url = f"{self.BASE_URL}/forecast?q={city}&appid={self.api_key}&units=metric&lang=ru&cnt=8"
        try:
            response = requests.get(url, timeout=15, headers=self.HEADERS)
            if response.status_code == 200:
                data = response.json()
                tomorrow = data['list'][4]  # примерно 15:00 следующего дня
                date = datetime.fromtimestamp(tomorrow['dt']).strftime('%d.%m.%Y')
                return (
                    f"🌆 Город: {data['city']['name']}\n"
                    f"📅 Прогноз на завтра ({date})\n"
                    f"⛅ Погода: {tomorrow['weather'][0]['description'].capitalize()}\n"
                    f"🌡 Температура: {tomorrow['main']['temp']:.1f}°C (ощущается как {tomorrow['main']['feels_like']:.1f}°C)\n"
                    f"💧 Влажность: {tomorrow['main']['humidity']}%\n"
                    f"🌬 Ветер: {tomorrow['wind']['speed']:.1f} м/с\n"
                    f"💧 Вероятность осадков: {tomorrow['pop'] * 100:.0f}%"
                )
            elif response.status_code == 404:
                return f"❌ Не могу найти город {city}"
            else:
                return "❌ Ошибка получения прогноза"
        except requests.exceptions.Timeout:
            return "❌ Превышено время ожидания ответа от сервиса погоды"
        except requests.exceptions.ConnectionError:
            return "❌ Не удалось подключиться к сервису погоды"