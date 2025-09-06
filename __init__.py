# Import all Modal services to ensure they get registered
from .unmute.modal_app import app
from .unmute.tts import orpheus_fastapi_app, orpheus_llama_app

# Export the main app for deployment
__all__ = ["app", "orpheus_fastapi_app", "orpheus_llama_app"]
