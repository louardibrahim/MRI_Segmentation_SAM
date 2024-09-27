from transformers import SamModel

def get_model():
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    # make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    return model
