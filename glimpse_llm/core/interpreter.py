from transformer_lens import HookedTransformer
from .server import InterpretabilityServer

class Interpreter:
    """Main interface for glimpse-llm"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize the interpreter with a model"""
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.to(device)
        self.server = InterpretabilityServer(self)
        
    def launch(self, port: int = 8000, open_browser: bool = True):
        """Launch the web interface"""
        self.server.launch(port=port, open_browser=open_browser)
    
    def analyze(self, text: str, analysis_types: list = None):
        """Run analysis on input text"""
        if analysis_types is None:
            analysis_types = ["attention", "activations"]
            
        results = {}
        tokens = self.model.tokenizer(text, return_tensors="pt")
        
        # Run model with hooks for capturing intermediate values
        with self.model.hooks():
            output = self.model(tokens)
            
            # Collect results based on requested analysis types
            if "attention" in analysis_types:
                results["attention"] = self.get_attention_patterns()
            if "activations" in analysis_types:
                results["activations"] = self.get_activations()
                
        return results