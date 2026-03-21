# Словарь ТОЛЬКО тех разговорных/сокращённых названий,
# которые pymorphy3 не приводит корректно к официальному названию города.
# Ключи всегда в нижнем регистре.

CITY_ALIASES: dict[str, str] = {
    # Москва — аббревиатура, pymorphy3 не знает
    "мск":                  "Moscow",

    # Санкт-Петербург — pymorphy3 знает "питер" и "спб" как Geox,
    # но не приводит к полному названию
    "спб":                  "Saint Petersburg",
    "спбг":                 "Saint Petersburg",
    "питер":                "Saint Petersburg",
    "петербург":            "Saint Petersburg",

    # Нижний Новгород — "нижний" pymorphy3 считает прилагательным (не Geox)
    "нижний":               "Nizhny Novgorod",
    "нижний новгород":      "Nizhny Novgorod",
    "нн":                   "Nizhny Novgorod",

    # Екатеринбург — "екб" pymorphy3 не знает совсем
    "екб":                  "Yekaterinburg",
    "екат":                 "Yekaterinburg",
    "ёбург":                "Yekaterinburg",
    "ебург":                "Yekaterinburg",

    # Новосибирск — pymorphy3 знает "новосиб" как Geox, но не приводит к полному
    "новосиб":              "Novosibirsk",
    "нск":                  "Novosibirsk",

    # Челябинск — pymorphy3 приводит "челяба" → "челяб" (неверно)
    "челяба":               "Chelyabinsk",
    "чел":                  "Chelyabinsk",

    # Владивосток — "владик" pymorphy3 не знает как Geox
    "владик":               "Vladivostok",

    # Хабаровск — "хаб" pymorphy3 не знает
    "хаб":                  "Khabarovsk",

    # Воронеж — сленговое сокращение
    "вронж":                "Voronezh",

    # Ростов-на-Дону — без уточнения OpenWeatherMap может найти не тот Ростов
    "ростов":               "Rostov-on-Don",
    "рнд":                  "Rostov-on-Don",

    # Великий Новгород — без "великий" найдёт Нижний
    "новгород":             "Veliky Novgorod",

    # Махачкала — сокращение
    "махач":                "Makhachkala",

    # Исторические названия (pymorphy3 знает слова, но не город)
    "ленинград":            "Saint Petersburg",
    "петроград":            "Saint Petersburg",
    "свердловск":           "Yekaterinburg",
    "горький":              "Nizhny Novgorod",
    "куйбышев":             "Samara",
    "сталинград":           "Volgograd",
    "царицын":              "Volgograd",
    "симбирск":             "Ulyanovsk",
}


def resolve_alias(city: str) -> str:
    """
    Возвращает официальное название города если передан алиас.
    Если алиас не найден — возвращает исходную строку без изменений.
    """
    return CITY_ALIASES.get(city.lower().strip(), city)