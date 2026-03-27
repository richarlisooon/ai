"""
intent_classifier.py
====================
Задача 2 (часть A): Загрузка обученной ML-модели и предсказание интента.

Подключается к боту вместо (или вместе с) regex-паттернами.
"""

import os
import logging
import joblib
from lemmatizer import LemmaTransformer  # noqa: F401 — нужен joblib при десериализации


# Метки интентов — совпадают с теми, что в dataset.csv
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

# Порог уверенности: если модель не уверена — возвращаем unknown
# LinearSVC не даёт вероятности, поэтому используем decision_function
CONFIDENCE_THRESHOLD = 0.0  # отключён: лучше вернуть слабый интент чем unknown


def _simple_preprocess(text: str) -> str:
    """Та же нормализация, что при обучении."""
    text = text.lower().strip()
    for ch in ".,!?;:\"'()-":
        text = text.replace(ch, " ")
    return " ".join(text.split())


class IntentClassifier:
    """
    Обёртка над sklearn Pipeline (TF-IDF + LinearSVC).

    Пример использования:
        clf = IntentClassifier()
        intent = clf.predict("какая погода в москве")  # → "weather"
    """

    def __init__(self, model_dir: str = "models", logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline = None
        self._load(model_dir)

    def _load(self, model_dir: str):
        model_path = os.path.join(model_dir, "model.pkl")
        if not os.path.exists(model_path):
            self.logger.error(
                f"❌ Файл модели не найден: {model_path}\n"
                f"   Сначала запустите: python train_model.py"
            )
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.pipeline = joblib.load(model_path)
        self.logger.info(f"✅ ML-модель загружена из {model_path}")
        print(f"✅ ML-модель интентов загружена")

    def predict(self, text: str) -> str:
        """
        Определяет интент текста.

        Возвращает одну из меток (INTENT_*) или INTENT_UNKNOWN,
        если уверенность модели ниже порога.
        """
        if not text or not text.strip():
            return INTENT_UNKNOWN

        processed = _simple_preprocess(text)

        try:
            # decision_function даёт «расстояние» до гиперплоскости —
            # чем больше, тем увереннее предсказание
            scores = self.pipeline.decision_function([processed])[0]
            predicted_label = self.pipeline.predict([processed])[0]

            # Для бинарного случая scores — scalar, для multi-class — массив
            if hasattr(scores, "__len__"):
                confidence = max(scores)
            else:
                confidence = abs(float(scores))

            self.logger.debug(
                f"IntentClassifier: текст={text!r} → {predicted_label} "
                f"(confidence={confidence:.3f})"
            )

            if confidence < CONFIDENCE_THRESHOLD:
                self.logger.debug("Уверенность ниже порога → unknown")
                return INTENT_UNKNOWN

            return predicted_label

        except Exception as e:
            self.logger.error(f"Ошибка предсказания интента: {e}")
            return INTENT_UNKNOWN

    def predict_with_score(self, text: str) -> tuple[str, float]:
        """Возвращает (intent, confidence) — удобно для отладки."""
        processed = _simple_preprocess(text)
        scores = self.pipeline.decision_function([processed])[0]
        label = self.pipeline.predict([processed])[0]
        confidence = float(max(scores)) if hasattr(scores, "__len__") else abs(float(scores))
        return label, confidence