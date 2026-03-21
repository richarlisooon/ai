import re
from datetime import datetime
from bot import Bot


def main():
    bot = Bot()
    print("боб готов! (с поддержкой NLP)")

    farewell_pattern = re.compile(
        r'\b(?:пока|до свидания|прощай|bye|goodbye|чао|увидимся|до встречи|покеда|всего доброго)\b',
        re.IGNORECASE
    )

    session_start = datetime.now()
    bot.logger.info(f"Новая сессия начата в {session_start}")

    try:
        while True:
            try:
                user_input = input("\nВы: ").strip()
                if not user_input:
                    continue

                response = bot.process_message(user_input)
                print("боб:", response)

                if farewell_pattern.search(user_input):
                    duration = datetime.now() - session_start
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
    finally:
        bot.close()


if __name__ == '__main__':
    main()