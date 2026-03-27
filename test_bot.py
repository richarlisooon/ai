"""
test_bot.py
===========
Проверка точности ML-классификатора на фразах,
которых НЕТ в датасете (реальные пользовательские запросы).

Запуск:
    python test_bot.py
"""

from intent_classifier import IntentClassifier

clf = IntentClassifier(model_dir="models")

TEST_CASES = [
    # greeting
    ("ну хай",                          "greeting"),
    ("добрейший день",                  "greeting"),
    ("хей как дела",                    "greeting"),
    ("о, привет",                       "greeting"),
    ("здравствуй дружище",              "greeting"),
    # goodbye
    ("ладно, я побежал",                "goodbye"),
    ("всё, спасибо, пока",              "goodbye"),
    ("до завтра дружище",               "goodbye"),
    ("мне уже пора",                    "goodbye"),
    ("окей, завершаем",                 "goodbye"),
    # weather
    ("на улице мокро?",                 "weather"),
    ("стоит ли брать зонтик",           "weather"),
    ("завтра будет солнце?",            "weather"),
    ("в питере сейчас холодно?",        "weather"),
    ("что за погода в сочи",            "weather"),
    # how_are_you
    ("ну ты как вообще",                "how_are_you"),
    ("как у тебя настроение",           "how_are_you"),
    ("бот, ты живой?",                  "how_are_you"),
    ("как дела бот",                    "how_are_you"),
    ("нормально себя чувствуешь",       "how_are_you"),
    # time
    ("скажи сколько щас",               "time"),
    ("время покажи пожалуйста",         "time"),
    ("хочу знать который час",          "time"),
    ("на часах сколько",                "time"),
    ("сейчас какое время",              "time"),
    # thanks
    ("спасибо тебе огромное",           "thanks"),
    ("очень признателен",               "thanks"),
    ("ты мне сильно помог, спс",        "thanks"),
    ("благодарочка",                    "thanks"),
    ("ты выручил, спасибо",             "thanks"),
    # compliment
    ("ты прям крутяк",                  "compliment"),
    ("боб ты огонь",                    "compliment"),
    ("ты вообще красавчик",             "compliment"),
    ("отработал на отлично",            "compliment"),
    ("ты реально топ",                  "compliment"),
    # who_are_you
    ("скажи кто ты",                    "who_are_you"),
    ("ты вообще кто",                   "who_are_you"),
    ("как тебя зовут бот",              "who_are_you"),
    ("ты живой или робот",              "who_are_you"),
    ("это вообще бот",                  "who_are_you"),
    # calculator
    ("посчитай мне 33 плюс 17",         "calculator"),
    ("сколько будет 9 на 9",            "calculator"),
    ("реши пример 100 минус 64",        "calculator"),
    ("вычисли 15 умножить на 4",        "calculator"),
    ("помоги решить пример",            "calculator"),
    # history
    ("напомни о чём мы говорили",       "history"),
    ("что ты мне отвечал раньше",       "history"),
    ("покажи наш диалог",               "history"),
    ("прошлые мои сообщения",           "history"),
    ("что было в начале разговора",     "history"),
]

def run_tests():
    print(f"{'Фраза':<45} {'Ожидалось':<15} {'Получилось':<15} {'OK?'}")
    print("-" * 85)

    correct = 0
    errors = []

    for phrase, expected in TEST_CASES:
        predicted, score = clf.predict_with_score(phrase)
        ok = predicted == expected
        if ok:
            correct += 1
        else:
            errors.append((phrase, expected, predicted, score))

        status = "OK" if ok else "!!"
        print(f"{phrase:<45} {expected:<15} {predicted:<15} {status}  (score={score:.2f})")

    total = len(TEST_CASES)
    accuracy = correct / total

    print("-" * 85)
    print(f"\nТочность: {correct}/{total} = {accuracy:.1%}")

    if errors:
        print(f"\nОшибки ({len(errors)}):")
        for phrase, expected, predicted, score in errors:
            print(f"  '{phrase}' -> ожидалось '{expected}', получилось '{predicted}' (score={score:.2f})")
    else:
        print("\nВсе фразы определены правильно!")

    print("\nТочность по интентам:")
    intents = sorted(set(e for _, e in TEST_CASES))
    for intent in intents:
        intent_cases = [(p, e) for p, e in TEST_CASES if e == intent]
        intent_correct = sum(1 for p, e in intent_cases if clf.predict(p) == e)
        bar = "#" * intent_correct + "." * (len(intent_cases) - intent_correct)
        print(f"  {intent:<15} [{bar}]  {intent_correct}/{len(intent_cases)}")

if __name__ == "__main__":
    run_tests()
