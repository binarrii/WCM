"""Safe response errors shared by model adapters and review reporting."""


class ModelResponseError(ValueError):
    def __init__(self, component: str, code: str, reason: str):
        super().__init__(reason)
        self.component = component
        self.code = code
        self.reason = reason
