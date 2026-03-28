"""
intent_classifier_spacy.py
===========================
Классификатор интентов на основе Word Embeddings (spaCy + LogisticRegression).

Подход: Текст → spaCy (doc.vector) → LogisticRegression.predict()

Используется в bot.py как основной классификатор (вместо TF-IDF + LinearSVC).
Требует предварительного запуска train_model_spacy.py.
"""

import os
import logging
import numpy as np
import joblib

# Метки интентов — совпадают с dataset.csv
INTENT_GREETING   = "greeting"
INTENT_GOODBYE    = "goodbye"
INTENT_WEATHER    = "weather"
INTENT_HOW        = "how_are_you"
INTENT_TIME       = "time"
INTENT_THANKS     = "thanks"
INTENT_COMPLIMENT = "compliment"
INTENT_WHO        = "who_are_you"
INTENT_CALCULATOR = "calculator"
INTENT_HISTORY    = "history"
INTENT_UNKNOWN    = "unknown"


class SpacyIntentClassifier:
    """
    Классификатор интентов на основе spaCy word embeddings.

    Пример использования:
        clf = SpacyIntentClassifier()
        intent = clf.predict("какая погода в москве")  # → "weather"
    """

    def __init__(self, model_dir: str = "models", logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.nlp = None
        self.model = None
        self._load(model_dir)

    def _load(self, model_dir: str):
        model_path = os.path.join(model_dir, "model_spacy.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"   Сначала запустите: python train_model_spacy.py"
            )

        # Загружаем spaCy только если есть модель (чтобы не тормозить при fallback)
        import spacy
        self.logger.info("Загрузка spaCy (ru_core_news_lg)...")
        self.nlp = spacy.load("ru_core_news_lg")

        self.model = joblib.load(model_path)
        self.logger.info(f"✅ spaCy-модель загружена из {model_path}")
        print("✅ spaCy ML-модель интентов загружена")

    def _vectorize(self, text: str) -> np.ndarray:
        """Получает вектор предложения через spaCy."""
        doc = self.nlp(text)
        return doc.vector

    def predict(self, text: str) -> str:
        """Определяет интент текста."""
        if not text or not text.strip():
            return INTENT_UNKNOWN
        try:
            X = self._vectorize(text).reshape(1, -1)
            return self.model.predict(X)[0]
        except Exception as e:
            self.logger.error(f"Ошибка предсказания spaCy-модели: {e}")
            return INTENT_UNKNOWN

    def predict_with_score(self, text: str) -> tuple[str, float]:
        """Возвращает (intent, вероятность) — удобно для отладки."""
        X = self._vectorize(text).reshape(1, -1)
        label = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = float(max(proba))
        return label, confidence
