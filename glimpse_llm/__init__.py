# glimpse_llm/__init__.py
from .core.interpreter import GlimpseInterpreter
from .core.server import GlimpseServer

class Glimpse:
    def __init__(self, model_name: str):
        self.interpreter = GlimpseInterpreter(model_name)
        self._server = None

    def launch(self, port: int = 8000):
        """Launch the web interface"""
        if not self._server:
            self._server = GlimpseServer(self.interpreter)
        self._server.start(port)

    def analyze(self, text: str):
        """Run analysis programmatically"""
        return self.interpreter.analyze_text(text)