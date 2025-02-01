MODEL_CONFIGS = {
   "gpt2": {
       "hooks": {
           "attention": {
               "module": "GPT2Attention",
               "output": 2
           },
           "activations": {
               "module": "GPT2MLP",
               "target": "act", 
               "type": "activation_fn"
           }
       }
   },
   "llama": {
       "hooks": {
           "attention": {
               "module": "LlamaAttention",
               "output": 0
           },
           "activations": {
               "module": "LlamaMLP",
               "target": "act_fn",
               "type": "activation_fn"
           }
       }
   }
}