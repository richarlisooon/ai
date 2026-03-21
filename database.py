import logging
import psycopg2
import config


class PostgresDB:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.conn = None
        self.cur = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(
                database=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=config.DB_PORT
            )
            self.conn.autocommit = True
            self.cur = self.conn.cursor()
            self.logger.info("База данных подключена!")
        except Exception as e:
            self.logger.error(f"Ошибка БД: {e}")
            raise e

    def save_user(self, name: str) -> int | None:
        try:
            self.cur.execute(
                "INSERT INTO users (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
                (name,)
            )
            result = self.cur.fetchone()
            if result:
                user_id = result[0]
            else:
                self.cur.execute("SELECT id FROM users WHERE name = %s", (name,))
                user_id = self.cur.fetchone()[0]

            self.conn.commit()
            self.logger.info(f"Пользователь {name} сохранен в БД")
            return user_id
        except Exception as e:
            self.logger.error(f"Ошибка сохранения пользователя: {e}")
            return None

    def save_log(self, user_id: int, message: str, response: str):
        if user_id:
            try:
                self.cur.execute(
                    "INSERT INTO logs (user_id, message, response) VALUES (%s, %s, %s)",
                    (user_id, message, response)
                )
                self.conn.commit()
            except Exception as e:
                self.logger.error(f"Ошибка сохранения лога: {e}")

    def save_nlp_query(self, user_id: int, original_query: str, analysis: dict | None, intent: str):
        try:
            uid = user_id if user_id else -1
            entities = str(analysis.get('entities', [])) if analysis else ''
            city = analysis.get('cities', [None])[0] if analysis and analysis.get('cities') else None

            self.cur.execute(
                """INSERT INTO processed_queries
                (user_id, original_query, processed_text, intent, entities, city)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (uid, original_query, str(analysis), intent, entities, city)
            )
            self.conn.commit()
            self.logger.info(f"NLP запрос сохранен: {intent}")
        except Exception as e:
            self.logger.error(f"Ошибка сохранения NLP запроса: {e}")

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()