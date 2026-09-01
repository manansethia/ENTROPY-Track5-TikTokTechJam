import argparse
import torch
from transformers import AutoModel

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
p.add_argument('--device',default='cpu')
a=p.parse_args()
dtype={'float32':torch.float32,'float16':torch.float16,'bfloat16':torch.bfloat16}[a.dtype]
model=AutoModel.from_pretrained(a.model, torch_dtype=dtype, device_map='cpu')
total=sum(x.numel() for x in model.parameters())
trainable=sum(x.numel() for x in model.parameters() if x.requires_grad)
print(f'model={a.model}')
print(f'total_parameters={total:,}')
print(f'trainable_parameters={trainable:,}')
print(f'under_2b={total < 2_000_000_000}')
