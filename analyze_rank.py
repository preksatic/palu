from loguru import logger
import torch.nn as nn
import torch
import os
import click
from tqdm import tqdm
from palu.data_utils import get_calib_data
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from utils import load_model_and_tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def numeric_rank(singular_values, rel_tol=1e-1):
    """
    Count of singular values greater than rel_tol * max(sv)
    """
    thresh = singular_values[0] * rel_tol
    return int((singular_values > thresh).sum().item())

@torch.no_grad()
def get_query_matrix(model, tokenizer, dev):
    model_id = model.config._name_or_path
    #NOTE (brian1009): Might need to check the random seed, currently we have < 0.1 perplexity difference at Llama2-7B
    calib_loader = get_calib_data(
        "wikitext2", 
        tokenizer, 
        model_id, 
        nsamples=32, 
        seqlen=2048
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    #FIXME: This is not a good implementation...
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id or "longchat":
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    query_matrices = []
    logger.info("[Decomposition] Start to calculate the q matrix in layer-wise manner...")

    head_dim = model.config.hidden_size // model.config.num_attention_heads

    rotary_emb = LlamaRotaryEmbedding(
            head_dim,
            max_position_embeddings=model.config.max_position_embeddings,
            base=model.config.rope_theta,
            )


    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            q_matrix = output.view(1, -1, model.config.num_attention_heads, head_dim).transpose(1, 2)
            #cos, sin = rotary_emb(q_matrix.cpu(), seq_len=q_matrix.shape[-2])
            #cos = cos.to(q_matrix.device)
            #sin = sin.to(q_matrix.device)
            #q_matrix, _ = apply_rotary_pos_emb(q_matrix, q_matrix, cos, sin, position_ids[0].unsqueeze(0))
            q_matrix = q_matrix[0][15]

            if module.query_matrix == None:
                module.query_matrix = q_matrix.detach().clone()
            else:
                module.query_matrix = torch.cat((module.query_matrix, q_matrix), dim = 0)
            del output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            if not ("q_proj" in name):
                continue
            subset[name].query_matrix = None
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        layer_query_matrices = {}
        for name in subset:
            if not ("q_proj" in name):
                continue
            layer_query_matrices[name] = subset[name].query_matrix.cpu()
            torch.cuda.empty_cache()
        query_matrices.append(layer_query_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()

    return query_matrices
        

if __name__ == "__main__":    
   
    model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-2-7b-hf")
    query_matrices = get_query_matrix(model, tokenizer, "cuda")
    sum = 0

    for layer_idx in range(len(query_matrices)):
        layer_query_matrices = query_matrices[layer_idx]
        for name in layer_query_matrices:
            q_matrix = layer_query_matrices[name]
            sv = torch.linalg.svdvals(q_matrix.float().to("cuda"))
            sv = sv.cpu()
            r = numeric_rank(sv)
            sum += r
            print(f"layer {layer_idx} head 15: {r}")
            torch.cuda.empty_cache()

    print(sum // 32)




