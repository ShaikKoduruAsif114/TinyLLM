import torch
from tokeniser.bpe_tokeniser import BPETokenizer
from model.gpt import GPT

config = {'vocab_size':4096,'n_embd':384,'n_head':6,'n_layer':6,'block_size':128,'dropout':0.1,'pos_encoding':'rope'}
model = GPT(config)

from pathlib import Path
VOCAB_PATH = Path('vocab.json')
MERGES_PATH = Path('merges.json')

tokenizer = BPETokenizer()
try:
    tokenizer.load(VOCAB_PATH, MERGES_PATH)
except Exception as e:
    print('Tokenizer load failed:', e)
    # create a small fake vocab to allow the script to run
    tokenizer.token_to_id = {'<PAD>':0,'<UNK>':1}
    tokenizer.id_to_token = {0:'<PAD>',1:'<UNK>'}
    tokenizer.merges = []
    token_ids = [1,1,1]
else:
    text = 'he is a cute little boy'
    token_ids = tokenizer.encode(text)

# prepare inputs
token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
embedded_tokens = model.token_embedding(token_tensor)
first_block = model.blocks[0]
normalized_embeddings = first_block.ln1(embedded_tokens)
first_attn = first_block.attn

model.eval()
qkv = first_attn.c_attn(normalized_embeddings)
q,k,v = qkv.split(config['n_embd'], dim=2)
B,T,C = q.shape
q = q.view(B,T,first_attn.n_head, first_attn.head_dim).transpose(1,2)
k = k.view(B,T,first_attn.n_head, first_attn.head_dim).transpose(1,2)
v = v.view(B,T,first_attn.n_head, first_attn.head_dim).transpose(1,2)
if getattr(first_attn,'rotary_emb',None) is not None:
    q,k = first_attn.rotary_emb(q,k)
att_scores = (q @ k.transpose(-2,-1)) * (1.0/(first_attn.head_dim**0.5))
mask = first_attn.bias[:, :, :T, :T]
att_scores = att_scores.masked_fill(mask==0, float('-inf'))
att_probs = torch.nn.functional.softmax(att_scores, dim=-1)
att_probs = first_attn.attn_dropout(att_probs)
y_manual = att_probs @ v
y_manual = y_manual.transpose(1,2).contiguous().view(B,T,C)
y_manual = first_attn.resid_dropout(first_attn.c_proj(y_manual))
module_output, module_att = first_attn(normalized_embeddings)
print('max diff', (y_manual-module_output).abs().max().item())
print('allclose', torch.allclose(y_manual, module_output, rtol=1e-5, atol=1e-6))
