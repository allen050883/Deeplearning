## Normalization Methods  
2D image = (N, H, W, C) --> (Batch size, Height, Weight, Channel)  
1. Batch Normalization: S = {K | K = C} --> calculate (N, H, W) to share on C  
2. Layer Normalization: S = {K | K = N} --> calculate (C, H, W) to share on N  
3. Instance Normalization: S = {K | K = (N, C)} --> calculate (H, W) to share on (N, C)  

## CNN attention
self-attention in CNN  
(Notice: cost lots of memory!!)  

## Reference  
1. WU, Yuxin; HE, Kaiming. Group normalization. arXiv preprint arXiv:1803.08494, 2018.
2. ZHANG, Han, et al. Self-attention generative adversarial networks. arXiv preprint arXiv:1805.08318, 2018.
