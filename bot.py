import re
import os
import random
from datetime import datetime

import config
from logger import setup_logging
from database import PostgresDB
from nlp import NLPEngine
from weather import WeatherService
from dialog_manager import DialogManager, DialogState
from intent_classifier import (
    IntentClassifier,
    INTENT_GREETING, INTENT_GOODBYE, INTENT_WEATHER,
    INTENT_HOW, INTENT_TIME, INTENT_THANKS, INTENT_COMPLIMENT,
    INTENT_WHO, INTENT_CALCULATOR, INTENT_HISTORY, INTENT_UNKNOWN,
)
from intent_classifier_spacy import SpacyIntentClassifier


class Bot:
    def __init__(self):
        self.patterns = []
        self.user_name = None
        self.user_id = None
        self.conversation_history = []

        self.logger = setup_logging()
        self.pg_db = PostgresDB(self.logger)
        self.nlp_engine = NLPEngine(self.logger)
        self.weather_service = WeatherService(config.WEATHER_API_KEY, self.logger)
        self.dialog = DialogManager()

        try:
            self.intent_clf = SpacyIntentClassifier(model_dir="models", logger=self.logger)
        except FileNotFoundError:
            self.logger.warning(
                "spaCy-модель не найдена — пробуем TF-IDF модель. "
                "Запустите train_model_spacy.py для обучения spaCy-модели."
            )
            try:
                self.intent_clf = IntentClassifier(model_dir="models", logger=self.logger)
            except FileNotFoundError:
                self.intent_clf = None
                self.logger.warning(
                    "ML-классификатор не загружен. "
                    "Запустите train_model_spacy.py или train_model.py."
                )

        self._register_patterns()
        self.logger.info("Бот инициализирован")

    def _register_patterns(self):
        self.logger.debug("Регистрация паттернов")

        self.patterns = [
            (re.compile(r'\b(?:очисти историю|забудь|clear history)\b', re.IGNORECASE), self.clear_history),
            (re.compile(r'\b(?:логи|покажи логи|посмотреть логи)\b', re.IGNORECASE), self.show_logs),
        ]

        self.calc_pattern = re.compile(r'([+-]?\d*\.?\d+)\s*([+\-*/])\s*([+-]?\d*\.?\d+)')

        self.name_pattern = re.compile(
            r'(?:меня зовут|my name is)\s+([А-Яа-яA-Za-z]+)|^(?:я)\s+([А-Яа-яA-Za-z]+)$',
            re.IGNORECASE
        )

        self.farewell_pattern = re.compile(
            r'\b(?:пока|до свидания|прощай|bye|goodbye|чао|увидимся|до встречи|покеда|всего доброго)\b',
            re.IGNORECASE
        )

    def _log_action(self, action: str, details: str = ""):
        user_info = f"Пользователь: {self.user_name or 'Неизвестный'}"
        self.logger.info(f"{user_info} - {action} - {details}")

    def _save_log(self, message: str, response: str):
        self.pg_db.save_log(self.user_id, message, response)

    def process_message(self, message: str) -> str:
        self.logger.info(f"Входящее сообщение: {message}")
        self.conversation_history.append(message)

        uid = self.user_id
        state = self.dialog.get_state(uid or 0)
        self.logger.info(f"Состояние диалога: {state}")

        nlp_analysis = self.nlp_engine.analyze(message)
        if nlp_analysis:
            self.logger.info(
                f"NLP анализ: сущности={nlp_analysis['entities']}, "
                f"города={nlp_analysis['cities']}, is_weather={nlp_analysis['is_weather_query']}"
            )

        if state == DialogState.WAIT_CITY:
            response = self._handle_wait_city(message, uid or 0, nlp_analysis)
            self._save_log(message, response)
            return response

        if state == DialogState.WAIT_DATE:
            response = self._handle_wait_date(message, uid or 0)
            self._save_log(message, response)
            return response

        name_match = self.name_pattern.search(message)
        if name_match:
            new_name = next((g for g in name_match.groups() if g is not None), None)
            if new_name:
                old_name = self.user_name
                self.user_name = new_name
                self.user_id = self.pg_db.save_user(self.user_name)
                self._log_action("представился", f"Был: {old_name}, Стал: {self.user_name}")
                response = f"приятно познакомиться, {self.user_name}!"
                self.logger.info(f"Исходящий ответ: {response}")
                self._save_log(message, response)
                return response

        calc_result = self._calculate(message)
        if calc_result:
            self.logger.info(f"Результат вычисления: {calc_result}")
            self._save_log(message, calc_result)
            return calc_result

        if self.intent_clf is not None:
            ml_intent = self.intent_clf.predict(message)
            self.logger.info(f"ML-классификатор: intent={ml_intent!r}")

            response = self._handle_ml_intent(message, ml_intent, uid or 0, nlp_analysis)
            if response is not None:
                self.logger.info(f"Исходящий ответ (ML): {response}")
                self._save_log(message, response)
                return response

        if self.farewell_pattern.search(message):
            response = self.farewell(self.farewell_pattern.search(message))
            self.logger.info(f"Исходящий ответ: {response}")
            self._save_log(message, response)
            return response

        if nlp_analysis and nlp_analysis.get('is_weather_query'):
            city_raw = self.nlp_engine.extract_city(message)
            if city_raw:
                response = self._weather_nlp(message, nlp_analysis, city_raw)
            else:
                forecast_words = ['завтра', 'на завтра', 'завтрашн', 'следующий день']
                is_forecast = any(w in message.lower() for w in forecast_words)
                self.dialog.set_context(uid or 0, 'is_forecast', is_forecast)
                self.dialog.set_state(uid or 0, DialogState.WAIT_CITY)
                response = "в каком городе?"
            self.logger.info(f"Исходящий ответ (NLP fallback): {response}")
            self._save_log(message, response)
            return response

        for pattern, handler in self.patterns:
            if pattern.search(message):
                response = handler(pattern.search(message))
                self.logger.info(f"Исходящий ответ: {response}")
                self._save_log(message, response)
                return response

        response = random.choice(["ниче не пон", "не понял, повтори", "че?", "а?", "ты кто", "говори проще"])
        self.logger.info(f"Ответ по умолчанию: {response}")
        self._log_action("непонятное сообщение", message)
        self._save_log(message, response)
        return response

    def _handle_ml_intent(self, message: str, intent: str, uid: int,
                          nlp_analysis: dict | None) -> str | None:
        if intent == INTENT_WEATHER:
            city_raw = self.nlp_engine.extract_city(message)
            if city_raw:
                return self._weather_nlp(message, nlp_analysis, city_raw)
            else:
                forecast_words = ["завтра", "на завтра", "завтрашн", "следующий день"]
                is_forecast = any(w in message.lower() for w in forecast_words)
                self.dialog.set_context(uid, "is_forecast", is_forecast)
                self.dialog.set_state(uid, DialogState.WAIT_CITY)
                return "в каком городе?"

        _fake_match = type("M", (), {"group": lambda self, n=0: message, "string": message})()

        if intent == INTENT_GREETING:
            return self.greeting(_fake_match)

        if intent == INTENT_GOODBYE:
            return self.farewell(_fake_match)

        if intent == INTENT_HOW:
            return self.how_are_you(_fake_match)

        if intent == INTENT_TIME:
            return self.time_response(_fake_match)

        if intent == INTENT_THANKS:
            return self.thanks_response(_fake_match)

        if intent == INTENT_COMPLIMENT:
            return self.compliment_response(_fake_match)

        if intent == INTENT_WHO:
            return self.who_are_you(_fake_match)

        if intent == INTENT_CALCULATOR:
            calc = self._calculate(message)
            if calc:
                return calc
            return self.calculator(_fake_match)

        if intent == INTENT_HISTORY:
            return self.show_history(_fake_match)

        return None

    def _handle_wait_city(self, message: str, uid: int, nlp_analysis: dict | None) -> str:
        city_raw = self.nlp_engine.extract_city(message)

        if not city_raw:
            city_raw = message.strip()

        city = self.nlp_engine.normalize_city(city_raw)
        is_forecast = self.dialog.get_context(uid, 'is_forecast') or False
        self.logger.info(f"WAIT_CITY → raw={city_raw!r}, город={city!r}, прогноз={is_forecast}")

        if is_forecast:
            self.dialog.set_context(uid, 'city', city)
            self.dialog.set_state(uid, DialogState.WAIT_DATE)
            return "на какую дату?"

        self.dialog.reset(uid)
        self.pg_db.save_nlp_query(uid, message, nlp_analysis, "weather")
        return self.weather_service.get_current(city)

    def _handle_wait_date(self, message: str, uid: int) -> str:
        city = self.dialog.get_context(uid, 'city') or config.DEFAULT_CITY
        self.logger.info(f"WAIT_DATE → город: {city}, дата: {message}")
        self.dialog.reset(uid)

        forecast_words = ['завтра', 'tomorrow']
        if any(w in message.lower() for w in forecast_words):
            return self.weather_service.get_forecast(city)

        return self.weather_service.get_current(city)

    def _weather_handler(self, match):
        return self._weather_nlp(match.string if hasattr(match, 'string') else '', None, None)

    def _weather_nlp(self, query_text: str, analysis: dict | None, city_raw: str | None = None) -> str:
        self._log_action("запрос погоды (NLP)")
        try:
            if not city_raw:
                city_raw = self.nlp_engine.extract_city(query_text)

            if city_raw:
                city = self.nlp_engine.normalize_city(city_raw)
                self.logger.info(f"Город: {city_raw} → {city}")
            else:
                city = config.DEFAULT_CITY
                self.logger.info(f"Город не найден, используем: {city}")

            if analysis is None:
                analysis = self.nlp_engine.analyze(query_text)

            forecast_words = ['завтра', 'на завтра', 'завтрашн', 'следующий день', 'на следующий день']
            is_forecast = any(word in query_text.lower() for word in forecast_words)

            intent = "weather_forecast" if is_forecast else "weather"
            self.pg_db.save_nlp_query(self.user_id, query_text, analysis, intent)

            return self.weather_service.get_forecast(city) if is_forecast else self.weather_service.get_current(city)
        except Exception as e:
            self.logger.error(f"Ошибка погоды: {e}")
            return "❌ Ошибка получения погоды"

    def _calculate(self, expression: str) -> str | None:
        try:
            match = self.calc_pattern.search(expression)
            if match:
                num1 = float(match.group(1))
                op = match.group(2)
                num2 = float(match.group(3))
                self._log_action("вычисление", f"{num1} {op} {num2}")

                if op == '+':
                    result = num1 + num2
                elif op == '-':
                    result = num1 - num2
                elif op == '*':
                    result = num1 * num2
                elif op == '/':
                    if num2 == 0:
                        self.logger.warning(f"Попытка деления на ноль: {num1} / 0")
                        return "на ноль делить нельзя!"
                    result = num1 / num2
                else:
                    return None

                return f"= {int(result)}" if result == int(result) else f"= {round(result, 2)}"
        except Exception as e:
            self.logger.error(f"Ошибка вычисления: {e}, выражение: {expression}")
        return None

    def greeting(self, match):
        self._log_action("приветствие", f"Текст: {match.group(0)}")
        if self.user_name:
            return random.choice([
                f"здарова, {self.user_name}!", f"привет, {self.user_name}!", f"здорово, {self.user_name}!"
            ])
        return "привет! как тебя зовут?"

    def farewell(self, match):
        self._log_action("прощание", f"Текст: {match.group(0)}")
        if self.user_name:
            return random.choice([
                f"пока, {self.user_name}!", f"до встречи, {self.user_name}!", f"всего доброго, {self.user_name}!"
            ])
        return random.choice(["покеда!", "до встречи!", "всего доброго!", "поки!", "до связи!"])

    def how_are_you(self, match):
        self._log_action("вопрос о делах")
        return random.choice(["нормуль", "отлично, а у тебя?", "с кайфом"])

    def time_response(self, match):
        current_time = datetime.now().strftime("%H:%M")
        self._log_action("запрос времени", f"Отправлено время: {current_time}")
        return f"щас {current_time}"

    def thanks_response(self, match):
        self._log_action("благодарность")
        return random.choice(["не за что!", "обращайся!", "пожалуйста!", "рад помочь!", "всегда пожалуйста!"])

    def compliment_response(self, match):
        self._log_action("комплимент")
        return random.choice(["спасибо, ты тоже!", "ой, спасибо!", "стараюсь!", "приятно слышать!"])

    def who_are_you(self, match):
        self._log_action("вопрос о личности бота")
        return "я боб"

    def calculator(self, match):
        self._log_action("запрос калькулятора")
        return "напиши пример (например: 2+2, 5*3, 10/2, 7-3)"

    def show_history(self, match):
        self._log_action("запрос истории")
        if len(self.conversation_history) < 2:
            return "история пустая"
        history = "последние сообщения:\n"
        for i, msg in enumerate(self.conversation_history[-5:], 1):
            history += f"{i}. {msg}\n"
        return history

    def clear_history(self, match):
        self._log_action("очистка истории")
        self.conversation_history.clear()
        return "история очищена"

    def show_logs(self, match):
        self._log_action("запрос логов")
        try:
            log_dir = "logs"
            log_files = os.listdir(log_dir)
            if not log_files:
                return "логов пока нет"
            latest_log = sorted(log_files)[-1]
            with open(os.path.join(log_dir, latest_log), 'r', encoding='utf-8') as f:
                lines = f.readlines()[-10:]
            logs = "последние 10 записей логов:\n"
            for line in lines:
                logs += line.strip() + "\n"
            return logs
        except Exception as e:
            self.logger.error(f"Ошибка при чтении логов: {e}")
            return "не могу прочитать логи"

    def close(self):
        self.pg_db.close()
        self.logger.info("Бот завершил работу, соединения закрыты")
