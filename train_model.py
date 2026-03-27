"""
train_model.py
==============
Обучение sklearn-классификатора с:
  - лемматизацией через pymorphy3
  - ансамблем TF-IDF (слова + символы)
  - подбором гиперпараметров через GridSearchCV
  - кросс-валидацией
  - экспортом модели
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
from lemmatizer import LemmaTransformer, lemmatize


# ──────────────────────── Загрузка данных ────────────────────────

def load_dataset(path):
    df = pd.read_csv(path).dropna(subset=['text','intent'])
    texts  = df['text'].tolist()
    labels = df['intent'].tolist()
    print(f'✅ Датасет: {len(texts)} примеров')
    dist = df['intent'].value_counts()
    for intent, cnt in dist.items():
        print(f'  {intent:20s}: {cnt}')
    return texts, labels


# ──────────────────────── Построение пайплайна ────────────────────────

def build_pipeline():
    """
    Pipeline:
      1. Лемматизация (pymorphy3)
      2. FeatureUnion из двух TF-IDF:
         - словесные n-граммы (1,2)
         - символьные n-граммы (2,4)  — устойчивы к опечаткам
      3. LinearSVC  (быстрый, точный для текстов)
    """
    word_tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=20000,
    )
    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True,
        max_features=30000,
    )
    features = FeatureUnion([
        ('word', word_tfidf),
        ('char', char_tfidf),
    ])
    pipeline = Pipeline([
        ('lemma', LemmaTransformer()),
        ('features', features),
        ('clf', LinearSVC(C=1.0, max_iter=3000, random_state=42)),
    ])
    return pipeline


# ──────────────────────── GridSearch ────────────────────────

def tune(pipeline, X_train, y_train):
    param_grid = {
        'clf__C': [0.1, 0.5, 1.0, 5.0, 10.0],
        'features__word__ngram_range': [(1,1),(1,2)],
        'features__char__ngram_range': [(2,4),(3,5)],
    }
    print('\n🔍 GridSearchCV (5-fold)...')
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print(f'  Лучшие параметры : {gs.best_params_}')
    print(f'  CV accuracy      : {gs.best_score_:.4f}')
    return gs.best_estimator_


# ──────────────────────── Обучение ────────────────────────

def train(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    print(f'\n📊 Train: {len(X_train)}, Test: {len(X_test)}')

    pipeline = build_pipeline()
    best = tune(pipeline, X_train, y_train)

    # Финальная оценка на тестовой выборке
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'\n🎯 Test accuracy: {acc:.4f}')
    print(classification_report(y_test, y_pred))

    # Кросс-валидация на полном датасете
    cv_scores = cross_val_score(best, texts, labels, cv=10, scoring='accuracy')
    print(f'📈 10-fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    print(f'   Минимум: {cv_scores.min():.4f}, Максимум: {cv_scores.max():.4f}')

    # Переобучаем на всём датасете после подбора гиперпараметров
    best.fit(texts, labels)
    print('\n✅ Финальная модель обучена на всём датасете')
    return best


# ──────────────────────── Экспорт ────────────────────────

def export_model(pipeline, out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(out_dir, 'model.pkl'))
    joblib.dump(pipeline.named_steps['features'].transformer_list[0][1],
                os.path.join(out_dir, 'vectorizer.pkl'))
    joblib.dump(pipeline.named_steps['clf'],
                os.path.join(out_dir, 'classifier.pkl'))
    print(f'💾 Сохранено в {out_dir}/')


# ──────────────────────── Smoke test ────────────────────────

def smoke_test(pipeline):
    tests = [
        ('привет',                    'greeting'),
        ('хаюшки',                    'greeting'),
        ('пока бот',                  'goodbye'),
        ('мне пора',                  'goodbye'),
        ('дождик будет?',             'weather'),
        ('холодно сегодня',           'weather'),
        ('как ты себя чувствуешь',    'how_are_you'),
        ('всё ок у тебя',             'how_are_you'),
        ('который час сейчас',        'time'),
        ('сколько на часах',          'time'),
        ('мерси большое',             'thanks'),
        ('ценю твою помощь',          'thanks'),
        ('топчик бот',                'compliment'),
        ('ты на высоте',              'compliment'),
        ('с кем я говорю',            'who_are_you'),
        ('ты нейросеть',              'who_are_you'),
        ('посчитай 12 плюс 8',        'calculator'),
        ('помоги с математикой',      'calculator'),
        ('покажи диалог',             'history'),
        ('что мы обсуждали',          'history'),
    ]
    print('\n🔍 Smoke-test:')
    correct = 0
    for phrase, expected in tests:
        pred = pipeline.predict([phrase])[0]
        ok = '✅' if pred == expected else '❌'
        if pred == expected:
            correct += 1
        print(f'  {ok} {phrase!r:40s} → {pred} (ожидалось: {expected})')
    print(f'\n  Smoke accuracy: {correct}/{len(tests)} = {correct/len(tests):.0%}')


# ──────────────────────── Main ────────────────────────

if __name__ == '__main__':
    texts, labels = load_dataset('dataset.csv')
    best = train(texts, labels)
    export_model(best)
    smoke_test(best)
    print('\n✅ Готово!')
