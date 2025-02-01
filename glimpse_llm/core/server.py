# glimpse_llm/server/server.py
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from typing import Dict, Set
import asyncio
from pathlib import Path

class GlimpseServer:
    def __init__(self, interpreter):
        self.app = FastAPI()
        self.interpreter = interpreter
        self.active_connections: Set[WebSocket] = set()
        
        # Setup routes
        self.setup_routes()
        
        # Serve static files (React frontend)
        static_dir = Path(__file__).parent.parent / "static"
        self.app.mount("/", StaticFiles(directory=static_dir, html=True))

    def setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)

        @self.app.get("/api/model/info")
        async def model_info():
            return {
                "name": self.interpreter.model.config.model_type,
                "num_layers": self.interpreter.model.config.num_hidden_layers,
                "num_heads": self.interpreter.model.config.num_attention_heads
            }

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        
        try:
            while True:
                message = await websocket.receive_json()
                
                if message["type"] == "analyze":
                    # Run analysis
                    results = self.interpreter.analyze_text(message["text"])
                    # Send results in chunks to avoid message size limits
                    await self.send_analysis_results(websocket, results)
                    
                elif message["type"] == "get_activation_details":
                    layer = message["layer"]
                    neuron = message["neuron"]
                    # Get detailed activation info for specific neuron
                    details = self.get_neuron_details(layer, neuron)
                    await websocket.send_json({
                        "type": "activation_details",
                        "data": details
                    })
                    
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.active_connections.remove(websocket)

    async def send_analysis_results(self, websocket: WebSocket, results: Dict):
        # Send each analysis type separately
        for analysis_type, data in results.items():
            await websocket.send_json({
                "type": f"analysis_{analysis_type}",
                "data": data
            })

    def get_neuron_details(self, layer: str, neuron: int) -> Dict:
        """Get detailed information about a specific neuron"""
        layer_data = self.interpreter.activation_cache.get(layer, {})
        if not layer_data:
            return {}
            
        activations = layer_data["activations"]
        neuron_activations = activations[:, :, neuron]
        
        return {
            "mean_activation": float(neuron_activations.mean().item()),
            "max_activation": float(neuron_activations.max().item()),
            "activation_pattern": neuron_activations.numpy().tolist()
        }

    def start(self, port: int = 8000):
        """Start the server"""
        uvicorn.run(self.app, host="0.0.0.0", port=port)