class VoiceUnavailable(RuntimeError):
    """A voice component cannot run on this machine right now (missing package, model or device).
    The message says exactly what to install or change; the daemon reports it and does NOT
    restart-loop, because retrying cannot fix a missing package."""
