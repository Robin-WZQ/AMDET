import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.set_seed import set_random_seed

class CustomCLIPTextEncoder(nn.Module):
    '''
    A custom text encoder class based on the CLIPTextModel from the Hugging Face Transformers library.
    '''
    def __init__(self, original_clip_text_encoder: CLIPTextModel):
        super().__init__()
        self.remaining_encoder_layers = original_clip_text_encoder.text_model.encoder.layers
        self.final_layer_norm = original_clip_text_encoder.text_model.final_layer_norm

        self.config = original_clip_text_encoder.config
        self.input_feature_dim = self.config.hidden_size
        
    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask
    
    # Copied from transformers.models.bart.modeling_bart._expand_mask
    def _expand_mask(self,mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len = hidden_states.shape[:2]

        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = self._expand_mask(attention_mask, hidden_states.dtype)

        for layer_module in self.remaining_encoder_layers:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask = causal_attention_mask,
            )
            hidden_states = layer_outputs[0]
            
        # Apply final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

class FirstLayerSimulator(nn.Module):
    def __init__(self, original_encoder: CLIPTextModel):
        super().__init__()
        # Access the embedding layer
        self.embeddings = original_encoder.text_model.embeddings

    def forward(self, 
                input_ids: torch.Tensor):         
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # Generate hidden states from the embeddings
        hidden_states = self.embeddings(input_ids=input_ids) # [batchsize, 77, 768]

        return hidden_states

# --- Example Usage ---
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_random_seed(42)

    # 1. Load the pre-trained CLIP model and tokenizer
    model_name = "/mnt/sdb1/wangzhongqi/project/backdoor_detection/T2IShield5.2/Models/Backdoor_Models/len2/poisoned_model_1"
    tokenizer = CLIPTokenizer.from_pretrained('/mnt/sdb1/wangzhongqi/Models/stable-diffusion-v1-4/tokenizer')
    original_text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)

    # 2. Assume we have a method to get the "output features of the first layer"
    first_layer_simulator = FirstLayerSimulator(original_text_encoder).to(device)

    # 3. Instantiate our custom text encoder (which includes all layers except the first layer)
    custom_encoder = CustomCLIPTextEncoder(original_text_encoder).to(device)

    # 4. Prepare input text
    texts = ["a photo of a cat", "a dog in the park"]
    inputs = tokenizer(texts, return_tensors="pt", padding="max_length",truncation=True, max_length=77)
    input_ids = inputs["input_ids"].to(device)

    # 5. Generate simulated "first layer output features" as input to CustomCLIPTextEncoder
    with torch.no_grad():
        hidden_states = first_layer_simulator(input_ids)

    print(f"\nInput text shape: {input_ids.shape}")
    print(f"Simulated first layer output feature shape: {hidden_states.shape}")

    # 6. Use the custom encoder for forward propagation
    with torch.no_grad():
        final_features = custom_encoder(hidden_states)

    print(f"\nCustomCLIPTextEncoder final output feature shape: {final_features.shape}")

    # 7. Verify if the custom encoder's output matches the original model's output
    with torch.no_grad():
        original_model_output = original_text_encoder(input_ids)
        original_last_hidden_state = original_model_output[0]
    
    print(f"Complete run of the original model's last_hidden_state shape: {original_last_hidden_state.shape}")
    
    print(f"Does the custom model output match the original model's last_hidden_state: "
          f"{torch.equal(final_features, original_last_hidden_state)}")