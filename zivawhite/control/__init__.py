"""Device control: a fixed command grammar (intents.py) plus per-OS backends (backends.py),
tied together by controller.py. Nothing in here imports torch."""
from .controller import DeviceController
from .intents import Intent, parse_intent
from .result import ActionResult

__all__ = ["Intent", "parse_intent", "ActionResult", "DeviceController"]
