"""
train_model_spacy.py
====================
Обучение классификатора интентов на основе Word Embeddings (spaCy).

Подход: Текст → spaCy → Word Embedding (doc.vector) → LogisticRegression

Запуск:
    pip install spacy
    python -m spacy download ru_core_news_md
    python train_model_spacy.py
"""

import os
import numpy as np
import pandas as pd
import spacy
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score


# ──────────────────────── Загрузка spaCy ────────────────────────

print("⏳ Загрузка модели spaCy (ru_core_news_lg)...")
nlp = spacy.load("ru_core_news_lg")
print("✅ spaCy загружен")


# ──────────────────────── Векторизация ────────────────────────

def vectorize(text: str) -> np.ndarray:
    """Получает вектор предложения через spaCy (среднее векторов слов)."""
    doc = nlp(text)
    return doc.vector


def vectorize_all(texts: list[str]) -> np.ndarray:
    """Векторизует список текстов."""
    return np.array([vectorize(text) for text in texts])


# ──────────────────────── Загрузка данных ────────────────────────

def load_dataset(path: str):
    df = pd.read_csv(path).dropna(subset=['text', 'intent'])
    texts = df['text'].tolist()
    labels = df['intent'].tolist()
    print(f'✅ Датасет: {len(texts)} примеров')
    dist = df['intent'].value_counts()
    for intent, cnt in dist.items():
        print(f'  {intent:20s}: {cnt}')
    return texts, labels


# ──────────────────────── Обучение ────────────────────────

def train(texts: list[str], labels: list[str]) -> LogisticRegression:
    print('\n⏳ Векторизация текстов через spaCy...')
    X = vectorize_all(texts)
    print(f'✅ Форма матрицы признаков: {X.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f'\n📊 Train: {len(X_train)}, Test: {len(X_test)}')

    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n🎯 Test accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

    # Кросс-валидация
    cv_scores = cross_val_score(model, X, labels, cv=5, scoring='accuracy')
    print(f'📈 5-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    # Переобучаем на полном датасете
    model.fit(X, labels)
    print('\n✅ Финальная модель обучена на всём датасете')
    return model


# ──────────────────────── Экспорт ────────────────────────

def export_model(model: LogisticRegression, out_dir: str = 'models'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'model_spacy.pkl')
    joblib.dump(model, path)
    print(f'💾 Сохранено: {path}')


# ──────────────────────── Smoke test ────────────────────────

def smoke_test(model: LogisticRegression):
    tests = [
        # ── greeting: стандарт ──
        ('привет',                                  'greeting'),
        ('здравствуй',                              'greeting'),
        ('добрый вечер',                            'greeting'),
        ('приветствую',                             'greeting'),
        # greeting: сленг / нестандарт
        ('хаюшки',                                  'greeting'),
        ('здарова бот',                             'greeting'),
        ('хей',                                     'greeting'),
        ('йоу',                                     'greeting'),
        ('ку',                                      'greeting'),
        # greeting: труднее (похоже на how_are_you)
        ('приветик как ты',                         'greeting'),

        # ── goodbye: стандарт ──
        ('пока',                                    'goodbye'),
        ('до свидания',                             'goodbye'),
        ('прощай',                                  'goodbye'),
        # goodbye: косвенные фразы
        ('мне пора идти',                           'goodbye'),
        ('я пошёл',                                 'goodbye'),
        ('ухожу',                                   'goodbye'),
        ('всё на сегодня',                          'goodbye'),
        ('заканчиваю разговор',                     'goodbye'),
        # goodbye: труднее (похоже на history)
        ('на сегодня всё спасибо',                  'goodbye'),

        # ── weather: стандарт ──
        ('какая погода сегодня',                    'weather'),
        ('дождик будет?',                           'weather'),
        ('холодно сегодня',                         'weather'),
        # weather: косвенные фразы
        ('нужна ли куртка',                         'weather'),
        ('стоит ли брать зонт',                     'weather'),
        ('будут ли осадки',                         'weather'),
        ('как там на улице',                        'weather'),
        ('тепло ли сейчас',                         'weather'),
        # weather: с городом
        ('погода в москве',                         'weather'),
        ('какой прогноз на завтра',                 'weather'),

        # ── how_are_you: стандарт ──
        ('как дела',                                'how_are_you'),
        ('как ты себя чувствуешь',                  'how_are_you'),
        ('как поживаешь',                           'how_are_you'),
        # how_are_you: сленг
        ('как ты там',                              'how_are_you'),
        ('всё норм у тебя',                         'how_are_you'),
        ('как сам',                                 'how_are_you'),
        ('что нового',                              'how_are_you'),

        # ── time: стандарт ──
        ('сколько времени',                         'time'),
        ('который час',                             'time'),
        ('сколько на часах',                        'time'),
        # time: косвенные
        ('скажи время',                             'time'),
        ('текущее время',                           'time'),
        # time: труднее (похоже на weather — "сейчас")
        ('что сейчас за время',                     'time'),

        # ── thanks: стандарт ──
        ('спасибо',                                 'thanks'),
        ('благодарю',                               'thanks'),
        ('спасибо большое',                         'thanks'),
        # thanks: сленг / иностранные слова
        ('мерси',                                   'thanks'),
        ('сенкс',                                   'thanks'),
        ('пасиб',                                   'thanks'),
        # thanks: труднее (похоже на compliment)
        ('ценю тебя',                               'thanks'),
        ('ты мне помог спасибо',                    'thanks'),

        # ── compliment: стандарт ──
        ('молодец',                                 'compliment'),
        ('отлично',                                 'compliment'),
        ('ты классный',                             'compliment'),
        # compliment: сленг
        ('топчик',                                  'compliment'),
        ('огонь',                                   'compliment'),
        ('красава',                                 'compliment'),
        ('бомба просто',                            'compliment'),
        # compliment: труднее (похоже на thanks)
        ('ты справился на отлично',                 'compliment'),

        # ── who_are_you: стандарт ──
        ('кто ты',                                  'who_are_you'),
        ('ты кто такой',                            'who_are_you'),
        ('как тебя зовут',                          'who_are_you'),
        # who_are_you: косвенные
        ('ты человек или бот',                      'who_are_you'),
        ('ты живой',                                'who_are_you'),
        ('ты робот',                                'who_are_you'),
        ('с кем я разговариваю',                    'who_are_you'),
        # who_are_you: труднее (похоже на greeting)
        ('представься пожалуйста',                  'who_are_you'),

        # ── calculator: стандарт ──
        ('посчитай 5 плюс 3',                       'calculator'),
        ('сколько будет 10 минус 4',                'calculator'),
        ('помоги посчитать',                        'calculator'),
        # calculator: косвенные
        ('реши пример',                             'calculator'),
        ('мне нужна математика',                    'calculator'),
        ('посчитай мне кое что',                    'calculator'),

        # ── history: стандарт ──
        ('покажи историю',                          'history'),
        ('что мы обсуждали',                        'history'),
        ('покажи диалог',                           'history'),
        # history: косвенные
        ('вернись к началу разговора',              'history'),
        ('напомни что я говорил',                   'history'),
        ('прокрути назад',                          'history'),
        ('предыдущие сообщения',                    'history'),
    ]
    print('\n🔍 Smoke-test:')
    correct = 0
    for phrase, expected in tests:
        vec = vectorize(phrase).reshape(1, -1)
        pred = model.predict(vec)[0]
        ok = '✅' if pred == expected else '❌'
        if pred == expected:
            correct += 1
        print(f'  {ok} {phrase!r:40s} → {pred} (ожидалось: {expected})')
    print(f'\n  Smoke accuracy: {correct}/{len(tests)} = {correct/len(tests):.0%}')


# ──────────────────────── Main ────────────────────────

if __name__ == '__main__':
    texts, labels = load_dataset('dataset.csv')
    model = train(texts, labels)
    export_model(model)
    smoke_test(model)
    print('\n✅ Готово! Запустите бота — он автоматически подхватит новую модель.')
