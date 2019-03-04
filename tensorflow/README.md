## Normalization Methods  
2D image = (N, H, W, C) --> (Batch size, Height, Weight, Channel)  
Batch Normalization: S = {K | K = C} --> calculate (N, H, W) to share on C  
Layer Normalization: S = {K | K = N} --> calculate (C, H, W) to share on N  
Instance Normalization: S = {K | K = (N, C)} --> calculate (H, W) to share on (N, C)  
  
## Reference  
1. WU, Yuxin; HE, Kaiming. Group normalization. arXiv preprint arXiv:1803.08494, 2018.
