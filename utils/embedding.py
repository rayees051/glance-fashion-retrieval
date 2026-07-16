import torch

def encode_query(query, model, tokenizer, device):

    tokens = tokenizer([query]).to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")