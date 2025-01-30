from .core.interpreter import Interpreter

__version__ = "0.1.0"

def launch(model_name="gpt2-small", port=8000):
    """Quick launch function for simple usage"""
    interpreter = Interpreter(model_name)
    interpreter.launch(port=port)