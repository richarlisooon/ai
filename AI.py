import re
import random
from datetime import datetime
import logging
import os
from logging.handlers import RotatingFileHandler
import requests
import psycopg2
import spacy
from spacy.matcher import Matcher
import sqlite3
import config

class Bot:
    def __init__(self):
        self.patterns = []
        self.user_name = None
        self.user_id = None
        self.conversation_history = []
        self.setup_logging()
        self.init_database()
        self.init_sqlite_db()
        self.init_spacy()
        self.register_patterns()
        self.logger.info("Бот инициализирован")
        self.weather_api_key = config.WEATHER_API_KEY

    def setup_logging(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filename = f"bot_log_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = os.path.join(log_dir, log_filename)

        self.logger = logging.getLogger('BotLogger')
        self.logger.setLevel(logging.DEBUG)

        file_handler = RotatingFileHandler(
            log_path, maxBytes=1048576, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def init_spacy(self):
        try:
            self.nlp = spacy.load("ru_core_news_sm")

            self.matcher = Matcher(self.nlp.vocab)

            weather_patterns = [
                [{"LEMMA": {"IN": ["погода", "прогноз"]}}],
                [{"LEMMA": "сколько"}, {"LEMMA": "градус"}],
                [{"LEMMA": "какой"}, {"LEMMA": "температура"}],
                [{"LEMMA": "холодно"}],
                [{"LEMMA": "тепло"}],
                [{"LEMMA": "дождь"}],
                [{"LEMMA": "солнечно"}],
                [{"LEMMA": "ветер"}],
                [{"LEMMA": "осадки"}],
            ]

            self.matcher.add("WEATHER_QUERY", weather_patterns)

            self.logger.info("spaCy модель успешно загружена")
            print("✅ spaCy модель загружена")

        except Exception as e:
            self.logger.error(f"Ошибка загрузки spaCy модели: {e}")
            print("❌ Ошибка загрузки spaCy модели. Убедитесь, что установлена ru_core_news_sm")
            raise e

    def init_sqlite_db(self):
        """Инициализация SQLite базы данных для хранения обработанных запросов"""
        try:
            self.sqlite_conn = sqlite3.connect('nlp_queries.db', check_same_thread=False)
            self.sqlite_cursor = self.sqlite_conn.cursor()

            self.sqlite_cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    original_query TEXT NOT NULL,
                    processed_text TEXT,
                    intent TEXT,
                    entities TEXT,
                    city TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.sqlite_conn.commit()
            self.logger.info("SQLite база данных инициализирована")
            print("✅ SQLite база данных подключена")

        except Exception as e:
            self.logger.error(f"Ошибка инициализации SQLite: {e}")
            raise e

    def init_database(self):
        try:
            self.conn = psycopg2.connect(
                database=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=config.DB_PORT
            )
            self.conn.autocommit = True

            self.cur = self.conn.cursor()

            self.logger.info("База данных подключена!")
            print("✅ База данных подключена")

        except Exception as e:
            self.logger.error(f"Ошибка БД: {e}")
            raise e

    def analyze_with_spacy(self, text):
        """Анализ текста с помощью spaCy"""
        try:
            doc = self.nlp(text)

            analysis = {
                'tokens': [token.text for token in doc],
                'lemmas': [token.lemma_ for token in doc],
                'pos_tags': [token.pos_ for token in doc],
                'entities': [],
                'cities': [],
                'is_weather_query': False
            }

            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                }
                analysis['entities'].append(entity_info)

                if ent.label_ in ["LOC", "GPE"]:
                    analysis['cities'].append(ent.text)

            matches = self.matcher(doc)
            if matches:
                analysis['is_weather_query'] = True

            weather_keywords = ['погод', 'прогноз', 'градус', 'температур', 'холодн', 'тепл', 'дожд', 'солнечн', 'ветр']
            if any(token.lemma_ in weather_keywords for token in doc):
                analysis['is_weather_query'] = True

            return analysis

        except Exception as e:
            self.logger.error(f"Ошибка spaCy анализа: {e}")
            return None

    def save_nlp_query(self, original_query, analysis, intent, response):
        """Сохранение обработанного NLP запроса в SQLite"""
        try:
            user_id = self.user_id if self.user_id else -1
            entities = str(analysis.get('entities', [])) if analysis else ''
            city = analysis.get('cities', [None])[0] if analysis and analysis.get('cities') else None

            self.sqlite_cursor.execute('''
                INSERT INTO processed_queries 
                (user_id, original_query, processed_text, intent, entities, city)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, original_query, str(analysis), intent, entities, city))

            self.sqlite_conn.commit()
            self.logger.info(f"NLP запрос сохранен в SQLite: {intent}")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения в SQLite: {e}")

    def save_user(self, name):
        try:
            self.cur.execute(
                "INSERT INTO users (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
                (name,)
            )
            result = self.cur.fetchone()
            if result:
                self.user_id = result[0]
            else:
                self.cur.execute("SELECT id FROM users WHERE name = %s", (name,))
                self.user_id = self.cur.fetchone()[0]

            self.conn.commit()
            self.logger.info(f"Пользователь {name} сохранен в БД")

        except Exception as e:
            self.logger.error(f"Ошибка сохранения пользователя: {e}")

    def save_log(self, message, response):
        if self.user_id:
            try:
                self.cur.execute(
                    "INSERT INTO logs (user_id, message, response) VALUES (%s, %s, %s)",
                    (self.user_id, message, response)
                )
                self.conn.commit()
            except Exception as e:
                self.logger.error(f"Ошибка сохранения лога: {e}")

    def register_patterns(self):
        self.logger.debug("Регистрация паттернов")

        self.patterns.append((re.compile(
            r'\b(?:привет|здравствуй|здравствуйте|хай|хелло|hello|hi|доброе утро|добрый день|добрый вечер|здарова|салют|хей)\b',
            re.IGNORECASE), self.greeting))
        self.patterns.append((re.compile(
            r'\b(?:пока|до свидания|прощай|bye|goodbye|чао|увидимся|до встречи|покеда|всего доброго)\b', re.IGNORECASE),
                              self.farewell))
        self.patterns.append(
            (re.compile(r'\b(?:как дела|как жизнь|как ты|чё как|how are you)\b', re.IGNORECASE), self.how_are_you))
        self.patterns.append(
            (re.compile(r'\b(?:сколько время|который час|времени)\b', re.IGNORECASE), self.time_response))
        self.patterns.append(
            (re.compile(r'\b(?:спасибо|благодарю|сенкс|thanks)\b', re.IGNORECASE), self.thanks_response))
        self.patterns.append((re.compile(r'\b(?:молодец|умница|класс|отлично|супер|красава)\b', re.IGNORECASE),
                              self.compliment_response))
        self.patterns.append((re.compile(r'\b(?:как тебя зовут|твоё имя|кто ты)\b', re.IGNORECASE), self.who_are_you))
        self.patterns.append(
            (re.compile(r'\b(?:калькулятор|посчитай|вычисли|сколько будет)\b', re.IGNORECASE), self.calculator))

        self.patterns.append(
            (re.compile(r'\b(?:погода|что на улице|холодно|тепло|температура|градусы)\b', re.IGNORECASE),
             self.weather_nlp))

        self.patterns.append(
            (re.compile(r'\b(?:история|что мы обсуждали|что я писал)\b', re.IGNORECASE), self.show_history))
        self.patterns.append(
            (re.compile(r'\b(?:очисти историю|забудь|clear history)\b', re.IGNORECASE), self.clear_history))
        self.patterns.append((re.compile(r'\b(?:логи|покажи логи|посмотреть логи)\b', re.IGNORECASE), self.show_logs))

        self.calc_pattern = re.compile(r'([+-]?\d*\.?\d+)\s*([+\-*/])\s*([+-]?\d*\.?\d+)')
        self.name_pattern = re.compile(r'(?:меня зовут|my name is)\s+([А-Яа-яA-Za-z]+)|^(?:я)\s+([А-Яа-яA-Za-z]+)$',
                                       re.IGNORECASE)

    def normalize_city_name(self, city_name):
        if not city_name:
            return None

        self.logger.info(f"Нормализация города: {city_name}")

        doc = self.nlp(city_name)

        normalized_parts = []

        for token in doc:
            if token.pos_ in ["PROPN", "NOUN"]:
                lemma = token.lemma_
                if lemma:
                    if '-' in lemma:
                        parts = lemma.split('-')
                        capitalized_parts = [part.capitalize() for part in parts]
                        normalized_parts.append('-'.join(capitalized_parts))
                    else:
                        normalized_parts.append(lemma.capitalize())
                else:
                    normalized_parts.append(token.text.capitalize())

            elif token.pos_ == "ADJ":
                lemma = token.lemma_
                if lemma:
                    normalized_parts.append(lemma.capitalize())
                else:
                    normalized_parts.append(token.text.capitalize())

            elif token.text in ['-']:
                normalized_parts.append(token.text)
            elif token.is_punct:
                continue
            else:
                if token.text.lower() not in ['в', 'во', 'на', 'под']:
                    normalized_parts.append(token.text)

        if normalized_parts:
            result = ' '.join(normalized_parts)
            result = ' '.join(result.split())
            self.logger.info(f"Нормализованное название: {result}")
            return result

        return city_name.title()

    def extract_city_from_text(self, text):
        if not text:
            return None

        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE"]:
                self.logger.info(f"NER нашел город: {ent.text}")
                return ent.text

        prepositions = ['в', 'во', 'на', 'под', 'над', 'из', 'до', 'для', 'около', 'возле', 'у']

        for i, token in enumerate(doc):
            if token.text.lower() in prepositions and i + 1 < len(doc):
                next_token = doc[i + 1]

                if next_token.pos_ in ["PROPN", "NOUN"] and next_token.text[0].isupper():
                    city_parts = [next_token.text]

                    j = i + 2
                    while j < len(doc) and (doc[j].pos_ in ["PROPN", "NOUN", "ADJ"] or doc[j].text in ['-']):
                        if doc[j].text[0].isupper() or doc[j].text in ['-']:
                            city_parts.append(doc[j].text)
                        else:
                            break
                        j += 1

                    city_name = ' '.join(city_parts)
                    self.logger.info(f"Контекст нашел город: {city_name}")
                    return city_name

        for token in doc:
            if token.pos_ == "PROPN" and token.text[0].isupper() and len(token.text) > 2:
                if token.i > 0 and doc[token.i - 1].text.lower() not in ['привет', 'здравствуй', 'пока']:
                    self.logger.info(f"Найдено потенциальное название города: {token.text}")
                    return token.text

        return None

    def weather_nlp(self, match):
        self.log_user_action("запрос погоды (NLP)")

        try:
            query_text = match.string if hasattr(match, 'string') else ''

            city_raw = self.extract_city_from_text(query_text)

            if city_raw:
                self.logger.info(f"Извлечен город: {city_raw}")
                city = self.normalize_city_name(city_raw)
            else:
                city = "Нижний Новгород"
                self.logger.info(f"Город не найден, используем: {city}")

            analysis = self.analyze_with_spacy(query_text)

            forecast_patterns = ['завтра', 'на завтра', 'завтрашн', 'следующий день', 'на следующий день']
            is_forecast = any(word in query_text.lower() for word in forecast_patterns)

            intent = "weather_forecast" if is_forecast else "weather"
            self.save_nlp_query(query_text, analysis, intent, city)

            if is_forecast:
                return self.get_weather_forecast(city)
            else:
                return self.get_current_weather(city)

        except Exception as e:
            self.logger.error(f"Ошибка погоды: {e}")
            return "❌ Ошибка получения погоды"

    def get_current_weather(self, city):
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.weather_api_key}&units=metric&lang=ru"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        try:
            response = requests.get(url, timeout=15, headers=headers)

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

    def get_weather_forecast(self, city):
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={self.weather_api_key}&units=metric&lang=ru&cnt=8"

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        try:
            response = requests.get(url, timeout=15, headers=headers)

            if response.status_code == 200:
                data = response.json()

                tomorrow_data = data['list'][4]  # примерно 15:00 следующего дня

                date = datetime.fromtimestamp(tomorrow_data['dt']).strftime('%d.%m.%Y')

                return (
                    f"🌆 Город: {data['city']['name']}\n"
                    f"📅 Прогноз на завтра ({date})\n"
                    f"⛅ Погода: {tomorrow_data['weather'][0]['description'].capitalize()}\n"
                    f"🌡 Температура: {tomorrow_data['main']['temp']:.1f}°C (ощущается как {tomorrow_data['main']['feels_like']:.1f}°C)\n"
                    f"💧 Влажность: {tomorrow_data['main']['humidity']}%\n"
                    f"🌬 Ветер: {tomorrow_data['wind']['speed']:.1f} м/с\n"
                    f"💧 Вероятность осадков: {tomorrow_data['pop'] * 100:.0f}%"
                )
            elif response.status_code == 404:
                return f"❌ Не могу найти город {city}"
            else:
                return "❌ Ошибка получения прогноза"
        except requests.exceptions.Timeout:
            return "❌ Превышено время ожидания ответа от сервиса погоды"
        except requests.exceptions.ConnectionError:
            return "❌ Не удалось подключиться к сервису погоды"


    def greeting(self, match):
        self.log_user_action("приветствие", f"Текст: {match.group(0)}")
        if self.user_name:
            response = random.choice([
                f"здарова, {self.user_name}!",
                f"привет, {self.user_name}!",
                f"здорово, {self.user_name}!"
            ])
        else:
            response = "привет! как тебя зовут?"

        self.logger.debug(f"Ответ на приветствие: {response}")
        return response

    def farewell(self, match):
        self.log_user_action("прощание", f"Текст: {match.group(0)}")
        if self.user_name:
            response = random.choice([
                f"пока, {self.user_name}!",
                f"до встречи, {self.user_name}!",
                f"всего доброго, {self.user_name}!"
            ])
        else:
            response = random.choice(["покеда!", "до встречи!", "всего доброго!", "поки!", "до связи!"])

        self.logger.info(f"Сессия завершена для {self.user_name if self.user_name else 'неизвестного пользователя'}")
        return response

    def how_are_you(self, match):
        self.log_user_action("вопрос о делах")
        return random.choice(["нормуль", "отлично, а у тебя?", "с кайфом"])

    def time_response(self, match):
        current_time = datetime.now().strftime("%H:%M")
        self.log_user_action("запрос времени", f"Отправлено время: {current_time}")
        return f"щас {current_time}"

    def thanks_response(self, match):
        self.log_user_action("благодарность")
        return random.choice(["не за что!", "обращайся!", "пожалуйста!", "рад помочь!", "всегда пожалуйста!"])

    def compliment_response(self, match):
        self.log_user_action("комплимент")
        return random.choice(["спасибо, ты тоже!", "ой, спасибо!", "стараюсь!", "приятно слышать!"])

    def who_are_you(self, match):
        self.log_user_action("вопрос о личности бота")
        return "я боб"

    def calculator(self, match):
        self.log_user_action("запрос калькулятора")
        return "напиши пример (например: 2+2, 5*3, 10/2, 7-3)"

    def weather(self, match):
        return self.weather_nlp(match)

    def show_history(self, match):
        self.log_user_action("запрос истории")
        if len(self.conversation_history) < 2:
            return "история пустая"

        history = "последние сообщения:\n"
        for i, msg in enumerate(self.conversation_history[-5:], 1):
            history += f"{i}. {msg}\n"
        return history

    def clear_history(self, match):
        self.log_user_action("очистка истории")
        self.conversation_history.clear()
        return "история очищена"

    def show_logs(self, match):
        self.log_user_action("запрос логов")
        try:
            log_dir = "logs"
            log_files = os.listdir(log_dir)
            if not log_files:
                return "логов пока нет"

            latest_log = sorted(log_files)[-1]
            log_path = os.path.join(log_dir, latest_log)

            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-10:]

            logs = "последние 10 записей логов:\n"
            for line in lines:
                logs += line.strip() + "\n"
            return logs
        except Exception as e:
            self.logger.error(f"Ошибка при чтении логов: {e}")
            return "не могу прочитать логи"

    def calculate(self, expression):
        try:
            match = self.calc_pattern.search(expression)
            if match:
                num1 = float(match.group(1))
                operator = match.group(2)
                num2 = float(match.group(3))

                self.log_user_action("вычисление", f"{num1} {operator} {num2}")

                if operator == '+':
                    result = num1 + num2
                elif operator == '-':
                    result = num1 - num2
                elif operator == '*':
                    result = num1 * num2
                elif operator == '/':
                    if num2 == 0:
                        self.logger.warning(f"Попытка деления на ноль: {num1} / 0")
                        return "на ноль делить нельзя!"
                    result = num1 / num2
                else:
                    return None

                if result.is_integer():
                    return f"= {int(result)}"
                else:
                    return f"= {round(result, 2)}"
            return None
        except Exception as e:
            self.logger.error(f"Ошибка вычисления: {e}, выражение: {expression}")
            return None

    def log_user_action(self, action, details=""):
        user_info = f"Пользователь: {self.user_name if self.user_name else 'Неизвестный'}"
        self.logger.info(f"{user_info} - {action} - {details}")

    def process_message(self, message):
        self.logger.info(f"Входящее сообщение: {message}")
        self.conversation_history.append(message)

        nlp_analysis = self.analyze_with_spacy(message)
        if nlp_analysis:
            self.logger.info(
                f"NLP анализ: сущности={nlp_analysis['entities']}, города={nlp_analysis['cities']}, is_weather={nlp_analysis['is_weather_query']}")

        farewell_pattern = re.compile(
            r'\b(?:пока|до свидания|прощай|bye|goodbye|чао|увидимся|до встречи|покеда|всего доброго)\b', re.IGNORECASE)
        if farewell_pattern.search(message):
            response = self.farewell(farewell_pattern.search(message))
            self.logger.info(f"Исходящий ответ: {response}")
            self.save_log(message, response)
            return response

        calc_result = self.calculate(message)
        if calc_result:
            self.logger.info(f"Результат вычисления: {calc_result}")
            self.save_log(message, calc_result)
            return calc_result

        if nlp_analysis and nlp_analysis.get('is_weather_query'):
            response = self.weather_nlp(re.search(r'.*', message))
            self.logger.info(f"Исходящий ответ (погода): {response}")
            self.save_log(message, response)
            return response

        name_match = self.name_pattern.search(message)
        if name_match:
            new_name = next((g for g in name_match.groups() if g is not None), None)
            if new_name:
                old_name = self.user_name
                self.user_name = new_name
                self.save_user(self.user_name)
                self.log_user_action("представился", f"Был: {old_name}, Стал: {self.user_name}")
                response = f"приятно познакомиться, {self.user_name}!"
                self.logger.info(f"Исходящий ответ: {response}")
                self.save_log(message, response)
                return response

        for pattern, handler in self.patterns:
            if pattern.search(message):
                response = handler(pattern.search(message))
                self.logger.info(f"Исходящий ответ: {response}")
                self.save_log(message, response)
                return response

        response = random.choice(["ниче не пон", "не понял, повтори", "че?", "а?", "ты кто", "говори проще"])
        self.logger.info(f"Ответ по умолчанию: {response}")
        self.log_user_action("непонятное сообщение", message)
        self.save_log(message, response)
        return response

    def __del__(self):
        if hasattr(self, 'cur'):
            self.cur.close()
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'sqlite_conn'):
            self.sqlite_conn.close()


if __name__ == '__main__':
    bot = Bot()
    print("боб готов! (с поддержкой NLP)")

    session_start = datetime.now()
    bot.logger.info(f"Новая сессия начата в {session_start}")
    while True:
        try:
            user_input = input("\nВы: ").strip()

            if not user_input:
                continue

            response = bot.process_message(user_input)
            print("боб:", response)

            farewell_pattern = re.compile(
                r'\b(?:пока|до свидания|прощай|bye|goodbye|чао|увидимся|до встречи|покеда|всего доброго)\b',
                re.IGNORECASE)
            if farewell_pattern.search(user_input):
                session_end = datetime.now()
                duration = session_end - session_start
                bot.logger.info(f"Сессия завершена. Длительность: {duration}")
                print("\nчат завершен")
                break

        except KeyboardInterrupt:
            bot.logger.warning("Сессия прервана пользователем (Ctrl+C)")
            print("\n\nчат прерван")
            break
        except Exception as e:
            bot.logger.error(f"Неожиданная ошибка: {e}")
            print(f"произошла ошибка: {e}")