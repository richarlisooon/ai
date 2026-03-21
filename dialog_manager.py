class DialogState:
    START = "start"
    WAIT_CITY = "wait_city"
    WAIT_DATE = "wait_date"


class DialogManager:
    def __init__(self):
        self.user_states: dict[int, str] = {}
        self.user_context: dict[int, dict] = {}  # временные данные диалога

    def get_state(self, user_id: int) -> str:
        if user_id in self.user_states:
            return self.user_states[user_id]
        return DialogState.START

    def set_state(self, user_id: int, state: str):
        self.user_states[user_id] = state

    def set_context(self, user_id: int, key: str, value):
        if user_id not in self.user_context:
            self.user_context[user_id] = {}
        self.user_context[user_id][key] = value

    def get_context(self, user_id: int, key: str):
        return self.user_context.get(user_id, {}).get(key)

    def clear_context(self, user_id: int):
        self.user_context.pop(user_id, None)

    def reset(self, user_id: int):
        self.set_state(user_id, DialogState.START)
        self.clear_context(user_id)