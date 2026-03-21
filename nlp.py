import logging
import pymorphy3
import spacy
from spacy.matcher import Matcher
from city_aliases import resolve_alias

_morph = pymorphy3.MorphAnalyzer()


def _to_nominative(word: str) -> str:
    """
    Приводит слово к именительному падежу через pymorphy3.
    Пример: "сарове" → "Саров", "арзамасе" → "Арзамас", "казани" → "Казань"
    """
    parsed = _morph.parse(word.lower())
    if not parsed:
        return word.capitalize()

    # Предпочитаем разбор с тегом Geox (топоним), иначе берём первый
    best = next((p for p in parsed if 'Geox' in p.tag), parsed[0])
    nom = best.inflect({'nomn'})
    if nom:
        return nom.word.capitalize()
    # Если не смог привести — возвращаем как есть с капитализацией
    return word.capitalize()


class NLPEngine:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.nlp = None
        self.matcher = None
        self._load_model()

    def _load_model(self):
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

    def analyze(self, text: str) -> dict | None:
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
                    analysis['cities'].append(resolve_alias(self._lemmatize_phrase(ent.text)))

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

    def extract_city(self, text: str) -> str | None:
        """Извлекает название города из текста, приводит к именительному + разворачивает алиасы."""
        if not text:
            return None

        # Сначала проверяем весь текст целиком как алиас ("спб", "екб", "нн", "питер")
        alias = resolve_alias(text.strip())
        if alias != text.strip():
            self.logger.info(f"Алиас всего текста: {text!r} → {alias!r}")
            return alias

        doc = self.nlp(text)

        # 1. NER — spaCy нашёл локацию
        for ent in doc.ents:
            if ent.label_ in ["LOC", "GPE"]:
                normalized = resolve_alias(self._lemmatize_phrase(ent.text))
                self.logger.info(f"NER нашел город: {ent.text!r} → {normalized!r}")
                return normalized

        # 2. Предлог + слово ("в москве", "в спб", "в нижнем")
        prepositions = {'в', 'во', 'на', 'из', 'до', 'около', 'возле', 'у', 'под', 'над', 'для'}
        for i, token in enumerate(doc):
            if token.text.lower() in prepositions and i + 1 < len(doc):
                next_tok = doc[i + 1]
                # Проверяем алиас сразу — "в нижнем" → "нижнем" → resolve → "Nizhny Novgorod"
                alias = resolve_alias(next_tok.text)
                if alias != next_tok.text:
                    self.logger.info(f"Алиас после предлога: {next_tok.text!r} → {alias!r}")
                    return alias
                # Иначе обычная обработка — принимаем любой POS кроме служебных
                if next_tok.pos_ not in ("ADP", "CCONJ", "SCONJ", "PUNCT", "NUM") and len(next_tok.text) > 1:
                    city_tokens = [next_tok]
                    j = i + 2
                    while j < len(doc):
                        t = doc[j]
                        if t.text == '-' or t.pos_ in ("PROPN", "NOUN", "ADJ"):
                            city_tokens.append(t)
                            j += 1
                        else:
                            break
                    raw = ' '.join(t.text for t in city_tokens)
                    normalized = resolve_alias(self._lemmatize_phrase(raw))
                    self.logger.info(f"Предлог нашел город: {raw!r} → {normalized!r}")
                    return normalized

        # 3. Одиночное собственное имя
        stop_words = {'привет', 'здравствуй', 'пока', 'боб', 'bob'}
        for token in doc:
            if (token.pos_ == "PROPN"
                    and len(token.text) > 2
                    and token.text.lower() not in stop_words):
                if token.i == 0 or doc[token.i - 1].text.lower() not in stop_words:
                    normalized = resolve_alias(self._lemmatize_token(token))
                    self.logger.info(f"PROPN нашел город: {token.text!r} → {normalized!r}")
                    return normalized

        return None

    def normalize_city(self, city_name: str) -> str:
        """Приводит название города к именительному падежу и разворачивает алиасы."""
        if not city_name:
            return city_name
        # Сначала проверяем алиас до лемматизации ("спб", "нн" и т.д.)
        alias_check = resolve_alias(city_name)
        if alias_check != city_name:
            return alias_check
        # Иначе лемматизируем и снова проверяем алиас
        lemmatized = self._lemmatize_phrase(city_name)
        return resolve_alias(lemmatized)

    def _lemmatize_token(self, token) -> str:
        """Приводит один токен к именительному падежу: spaCy → pymorphy3."""
        spacy_lemma = token.lemma_.strip()
        if spacy_lemma and spacy_lemma.lower() != token.text.lower():
            return spacy_lemma.capitalize()
        return _to_nominative(token.text)

    def _lemmatize_phrase(self, phrase: str) -> str:
        """
        Приводит фразу-название города к именительному падежу.
        spaCy используется для разбора структуры фразы,
        pymorphy3 — для приведения каждого слова к именительному.

        Примеры:
          "в сарове"           → "Саров"
          "арзамасе"           → "Арзамас"
          "казани"             → "Казань"
          "нижнем новгороде"   → "Нижний Новгород"
          "ростове-на-дону"    → "Ростов-на-Дону"
        """
        if not phrase:
            return phrase

        phrase_titled = ' '.join(w.capitalize() for w in phrase.split())
        doc = self.nlp(phrase_titled)
        parts = []

        for token in doc:
            if token.is_punct and token.text != '-':
                continue
            if token.text == '-':
                parts.append('-')
                continue

            if token.pos_ in ("PROPN", "NOUN", "ADJ"):
                spacy_lemma = token.lemma_.strip()
                # Если spaCy дал нормальную лемму — берём её
                if spacy_lemma and spacy_lemma.lower() != token.text.lower():
                    if '-' in spacy_lemma:
                        parts.append('-'.join(p.capitalize() for p in spacy_lemma.split('-')))
                    else:
                        parts.append(spacy_lemma.capitalize())
                else:
                    # Fallback: pymorphy3 точно приведёт к именительному
                    parts.append(_to_nominative(token.text))
            else:
                # Служебные слова ("на" в "Ростов-на-Дону")
                parts.append(token.text.lower())

        if not parts:
            return phrase_titled

        result = ' '.join(parts)
        result = result.replace(' - ', '-').strip()
        result = ' '.join(result.split())
        self.logger.info(f"Лемматизация фразы: {phrase!r} → {result!r}")
        return result

    def normalize_city(self, city_name: str) -> str:
        """
        Финальная нормализация: если название уже в именительном падеже
        (после extract_city), просто проверяем капитализацию.
        Если нет — прогоняем через лемматизацию.
        """
        if not city_name:
            return city_name
        return self._lemmatize_phrase(city_name)