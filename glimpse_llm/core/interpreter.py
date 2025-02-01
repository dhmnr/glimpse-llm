# glimpse_llm/core/interpreter.py
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional
import torch.nn.functional as F

from ..config import MODEL_CONFIGS

class GlimpseInterpreter:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.activation_cache = {}
        self.attention_cache = {}
        self._setup_hooks()
        
    def _setup_hooks(self):
    
        model_config = MODEL_CONFIGS[self.model_name]["hooks"]
        def make_hook(hook_type, config):
            def hook(module, input, output):
                if hook_type == "attention":
                    pattern = output[config["output"]]
                    self.attention_cache[f"attention_{len(self.attention_cache)}"] = {
                        "pattern": pattern.detach().cpu(),
                        "heads": pattern.shape[1]
                    }
                elif hook_type == "activations":
                    self.activation_cache[f"layer_{len(self.activation_cache)}"] = {
                        "activations": output.detach().cpu(),
                        "neurons": output.shape[-1]
                    }
                return output
            return hook

        for name, module in self.model.named_modules():
            for hook_type, config in model_config.items():
                if config["module"] in str(type(module).__name__):
                    if "target" in config:
                        # Hook specific attribute (like activation function)
                        target = getattr(module, config["target"])
                        target.register_forward_hook(make_hook(hook_type, config))
                    else:
                        # Hook entire module
                        module.register_forward_hook(make_hook(hook_type, config))

    def analyze_text(self, text: str) -> Dict:
        """Main analysis method"""
        self.activation_cache = {}
        self.attention_cache = {}
        
        text = text.strip()

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        clean_tokens = [self._clean_token(token) for token in tokens]
        analysis = {
            "activations": self._analyze_activations(),
            "attention": self._analyze_attention(clean_tokens),
            "circuits": self._discover_circuits(),
            "features": self._analyze_features(),
            "tokens": clean_tokens
        }
        
        return analysis

    def _clean_token(self, token: str) -> str:
        """Clean token by removing special characters and handling spaces"""
        if token.startswith('Ġ'):
            return ' ' + token[1:]
        return token

    def _analyze_activations(self) -> Dict:
        results = {}
        for layer_name, data in self.activation_cache.items():
            activations = data["activations"]
            
            # Basic statistics
            stats = {
                "mean": activations.mean(dim=1).numpy().tolist(),
                "max": activations.max(dim=1).values.numpy().tolist(),
                "sparsity": (activations == 0).float().mean().item(),
            }
            
            # Neuron importance
            importance = torch.norm(activations, dim=1)
            top_k = 10
            top_values, top_indices = torch.topk(importance, k=min(top_k, importance.shape[-1]))
            
            # Activation patterns
            patterns = {
                "top_neurons": {
                    "indices": top_indices.numpy().tolist(),
                    "values": top_values.numpy().tolist()
                },
                "activation_map": activations.mean(dim=1).numpy().tolist()
            }
            
            results[layer_name] = {
                "statistics": stats,
                "patterns": patterns,
                "dimensions": list(activations.shape)
            }
            
        return results

    def _analyze_attention(self, tokens: List[str]) -> Dict:
        results = {}
        for layer_name, data in self.attention_cache.items():
            pattern = data["pattern"]
            n_heads = pattern.shape[1]
            
            # Head importance scores
            head_importance = torch.norm(pattern, dim=(-2, -1))
            
            # Attention patterns per head
            head_patterns = pattern.mean(dim=0).numpy()  # Average over batch
            
            # Head clustering
            head_similarities = self._compute_head_similarities(pattern)
            head_clusters = self._cluster_heads(head_similarities)
            
            results[layer_name] = {
                "head_importance": head_importance.numpy().tolist(),
                "attention_patterns": head_patterns.tolist(),
                "head_clusters": head_clusters,
                "num_heads": n_heads,
                "tokens": tokens
            }
            
        return results

    def _discover_circuits(self) -> List[Dict]:
        circuits = []
        
        for layer_idx, (layer_name, layer_data) in enumerate(self.activation_cache.items()):
            activations = layer_data["activations"]
            
            # Compute neuron correlations
            corr_matrix = self._compute_neuron_correlations(activations)
            
            # Find clusters of correlated neurons
            clusters = self._cluster_neurons(corr_matrix)
            
            # Get corresponding attention patterns
            attn_name = f"attention_{layer_idx}"
            attn_data = self.attention_cache.get(attn_name)
            
            if attn_data:
                attn_pattern = attn_data["pattern"]
                
                # Associate neurons with attention heads
                for cluster_idx, neuron_cluster in enumerate(clusters):
                    circuit = {
                        "id": f"circuit_{layer_idx}_{cluster_idx}",
                        "layer": layer_idx,
                        "neurons": neuron_cluster,
                        "type": self._determine_circuit_type(
                            activations[:, :, neuron_cluster], 
                            attn_pattern
                        ),
                        "strength": float(
                            torch.norm(activations[:, :, neuron_cluster]).item()
                        )
                    }
                    circuits.append(circuit)
        
        return circuits

    def _analyze_features(self) -> Dict:
        features = {}
        
        for layer_name, data in self.activation_cache.items():
            activations = data["activations"]
            
            # Feature importance based on activation magnitude
            importance = torch.norm(activations, dim=1).mean(dim=0)
            
            # Feature correlations
            correlations = self._compute_feature_correlations(activations)
            
            # Feature clustering
            clusters = self._cluster_features(correlations)
            
            features[layer_name] = {
                "importance": importance.numpy().tolist(),
                "correlations": correlations.numpy().tolist(),
                "clusters": clusters
            }
            
        return features

    def _compute_neuron_correlations(self, activations: torch.Tensor) -> torch.Tensor:
        # Reshape to 2D: (batch * seq_len, neurons)
        acts_2d = activations.reshape(-1, activations.shape[-1])
        
        # Compute correlation matrix
        acts_centered = acts_2d - acts_2d.mean(dim=0, keepdim=True)
        cov = acts_centered.T @ acts_centered
        norm = torch.sqrt(torch.diag(cov).reshape(-1, 1) @ torch.diag(cov).reshape(1, -1))
        corr = cov / (norm + 1e-8)
        
        return corr

    def _cluster_neurons(self, correlation_matrix: torch.Tensor, threshold: float = 0.5) -> List[List[int]]:
        # Simple clustering based on correlation threshold
        clusters = []
        used_neurons = set()
        
        for i in range(correlation_matrix.shape[0]):
            if i in used_neurons:
                continue
                
            # Find correlated neurons
            correlated = torch.where(correlation_matrix[i] > threshold)[0].tolist()
            
            if len(correlated) > 1:  # Only create clusters with multiple neurons
                clusters.append(correlated)
                used_neurons.update(correlated)
                
        return clusters

    def _compute_head_similarities(self, attention_patterns: torch.Tensor) -> torch.Tensor:
        # Reshape to 2D: (batch * seq_len * seq_len, heads)
        patterns_2d = attention_patterns.reshape(-1, attention_patterns.shape[1])
        
        # Compute cosine similarity between heads
        similarities = F.cosine_similarity(
            patterns_2d.unsqueeze(1),
            patterns_2d.unsqueeze(0),
            dim=2
        )
        
        return similarities

    def _cluster_heads(self, similarities: torch.Tensor, threshold: float = 0.7) -> List[List[int]]:
        clusters = []
        used_heads = set()
        
        for i in range(similarities.shape[0]):
            if i in used_heads:
                continue
                
            similar_heads = torch.where(similarities[i] > threshold)[0].tolist()
            
            if len(similar_heads) > 1:
                clusters.append(similar_heads)
                used_heads.update(similar_heads)
                
        return clusters

    def _determine_circuit_type(
        self, 
        neuron_activations: torch.Tensor, 
        attention_patterns: torch.Tensor
    ) -> str:
        # Simple heuristic for circuit type classification
        if torch.norm(attention_patterns).item() > torch.norm(neuron_activations).item():
            return "Attention"
        else:
            return "MLP"

    def _compute_feature_correlations(self, activations: torch.Tensor) -> torch.Tensor:
        # Similar to neuron correlations but for feature dimensions
        acts_2d = activations.transpose(-1, -2).reshape(activations.shape[-1], -1)
        
        acts_centered = acts_2d - acts_2d.mean(dim=1, keepdim=True)
        cov = acts_centered @ acts_centered.T
        norm = torch.sqrt(torch.diag(cov).reshape(-1, 1) @ torch.diag(cov).reshape(1, -1))
        corr = cov / (norm + 1e-8)
        
        return corr

    def _cluster_features(self, correlations: torch.Tensor) -> List[List[int]]:
        # Similar to neuron clustering but for features
        clusters = []
        used_features = set()
        threshold = 0.5
        
        for i in range(correlations.shape[0]):
            if i in used_features:
                continue
                
            correlated = torch.where(correlations[i] > threshold)[0].tolist()
            
            if len(correlated) > 1:
                clusters.append(correlated)
                used_features.update(correlated)
                
        return clusters