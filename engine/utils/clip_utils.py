import torch
import open_clip

def load_openclip_text_features(
    class_names,
    model_name='ViT-B-32',
    pretrained='laion2b_s34b_b79k',
    device='cuda',
    prompt_template="a photo of a {}"
):
    """
    Load CLIP text features for a list of class names using OpenCLIP.

    Args:
        class_names (List[str]): List of class names.
        model_name (str): OpenCLIP model architecture name.
        pretrained (str): Pretrained weights identifier.
        device (str): Device to use ('cuda' or 'cpu').
        prompt_template (str): Prompt template for text description.

    Returns:
        torch.Tensor: Normalized CLIP text embeddings of shape [num_classes, clip_dim].
    """
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.eval()

    prompts = [prompt_template.format(c) for c in class_names]
    tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features  # shape: [num_classes, clip_dim]
