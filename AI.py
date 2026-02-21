import re
import random
from datetime import datetime
import logging
import os
from logging.handlers import RotatingFileHandler


class Bot:
    def __init__(self):
        self.patterns = []
        self.user_name = None
        self.conversation_history = []
        self.setup_logging()
        self.register_patterns()
        self.logger.info("Бот инициализирован")

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

    def log_user_action(self, action, details=""):
        user_info = f"Пользователь: {self.user_name if self.user_name else 'Неизвестный'}"
        self.logger.info(f"{user_info} - {action} - {details}")

    def register_patterns(self):
        self.logger.debug("Регистрация паттернов")

        self.patterns.append((
            re.compile(
                r'\b(?:привет|здравствуй|здравствуйте|хай|хелло|hello|hi|доброе утро|добрый день|добрый вечер|здарова|салют|хей)\b',
                re.IGNORECASE),
            self.greeting
        ))

        self.patterns.append((
            re.compile(r'\b(?:пока|до свидания|прощай|bye|goodbye|чао|увидимся|до встречи|покеда|всего доброго)\b',
                       re.IGNORECASE),
            self.farewell
        ))

        self.patterns.append((
            re.compile(r'\b(?:как дела|как жизнь|как ты|чё как|how are you)\b', re.IGNORECASE),
            self.how_are_you
        ))

        self.patterns.append((
            re.compile(r'\b(?:сколько время|который час|времени)\b', re.IGNORECASE),
            self.time_response
        ))

        self.patterns.append((
            re.compile(r'\b(?:спасибо|благодарю|сенкс|thanks)\b', re.IGNORECASE),
            self.thanks_response
        ))

        self.patterns.append((
            re.compile(r'\b(?:молодец|умница|класс|отлично|супер|красава)\b', re.IGNORECASE),
            self.compliment_response
        ))

        self.patterns.append((
            re.compile(r'\b(?:как тебя зовут|твоё имя|кто ты)\b', re.IGNORECASE),
            self.who_are_you
        ))

        self.patterns.append((
            re.compile(r'\b(?:калькулятор|посчитай|вычисли|сколько будет)\b', re.IGNORECASE),
            self.calculator
        ))

        self.patterns.append((
            re.compile(r'\b(?:погода|что на улице|холодно|тепло)\b', re.IGNORECASE),
            self.weather
        ))

        self.patterns.append((
            re.compile(r'\b(?:история|что мы обсуждали|что я писал)\b', re.IGNORECASE),
            self.show_history
        ))

        self.patterns.append((
            re.compile(r'\b(?:очисти историю|забудь|clear history)\b', re.IGNORECASE),
            self.clear_history
        ))

        self.patterns.append((
            re.compile(r'\b(?:логи|покажи логи|посмотреть логи)\b', re.IGNORECASE),
            self.show_logs
        ))

        self.calc_pattern = re.compile(r'([+-]?\d*\.?\d+)\s*([+\-*/])\s*([+-]?\d*\.?\d+)')
        self.name_pattern = re.compile(r'(?:меня зовут|my name is|я )\s*([А-Яа-яA-Za-z]+)', re.IGNORECASE)

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
        self.log_user_action("запрос погоды")
        return ("седня не")

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

    def process_message(self, message):
        self.logger.info(f"Входящее сообщение: {message}")
        self.conversation_history.append(message)

        farewell_pattern = re.compile(
            r'\b(?:пока|до свидания|прощай|bye|goodbye|чао|увидимся|до встречи|покеда|всего доброго)\b', re.IGNORECASE)
        if farewell_pattern.search(message):
            response = self.farewell(farewell_pattern.search(message))
            self.logger.info(f"Исходящий ответ: {response}")
            return response

        calc_result = self.calculate(message)
        if calc_result:
            self.logger.info(f"Результат вычисления: {calc_result}")
            return calc_result

        name_match = self.name_pattern.search(message)
        if name_match and not re.search(r'\b(?:как тебя зовут|твоё имя|кто ты)\b', message, re.IGNORECASE):
            old_name = self.user_name
            self.user_name = name_match.group(1)
            self.log_user_action("представился", f"Был: {old_name}, Стал: {self.user_name}")
            response = f"приятно познакомиться, {self.user_name}!"
            self.logger.info(f"Исходящий ответ: {response}")
            return response

        for pattern, handler in self.patterns:
            if pattern.search(message):
                response = handler(pattern.search(message))
                self.logger.info(f"Исходящий ответ: {response}")
                return response

        response = random.choice(["ниче не пон", "не понял, повтори", "че?", "а?", "ты кто", "говори проще"])
        self.logger.info(f"Ответ по умолчанию: {response}")
        self.log_user_action("непонятное сообщение", message)
        return response


if __name__ == '__main__':
    bot = Bot()
    print("боб готов!")

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