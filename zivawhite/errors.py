"""Exceptions shared across layers. Torch-free and framework-free on purpose, so the assistant, the router and the
web server can all catch the same classes without importing each other."""


class ModelBusy(RuntimeError):
    """The model is generating for another request and did not free up within the lock timeout. The web server
    answers 503 (try again shortly); the assistant tells the person it is busy."""
