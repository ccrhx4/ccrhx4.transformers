from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

device = "hpu"

if device == "hpu":
    import habana_frameworks.torch as ht

weight = torch.load("self_attn.q_proj.weight.pt")
bias = torch.load("self_attn.q_proj.bias.pt")
hidden_states = torch.load("token.0.layer.0.input_layernorm.hpu.pt")

hidden_states = hidden_states.to(device)
weight = weight.to(device)
bias = bias.to(device)

print("weight dtype: ", weight.dtype)
print("bias dtype: ", bias.dtype)
print("input dtype: ", hidden_states.dtype)

linear_q_proj = F.linear(hidden_states, weight, bias)

print("Number of zero(s) in Linear output:", (linear_q_proj.numel() - torch.count_nonzero(linear_q_proj)))

