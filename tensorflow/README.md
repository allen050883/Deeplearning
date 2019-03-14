## 1. Normalization Methods  
2D image = (N, H, W, C) --> (Batch size, Height, Weight, Channel)  
1. Batch Normalization: S = {K | K = C} --> calculate (N, H, W) to share on C  
2. Layer Normalization: S = {K | K = N} --> calculate (C, H, W) to share on N  
3. Instance Normalization: S = {K | K = (N, C)} --> calculate (H, W) to share on (N, C)  

## 2. CNN attention
self-attention in CNN  
(Notice: cost lots of memory!!)  

## 3. Coordinate Convolution
add coordinate information to model

## 4. Feature Pyramid Network
1. find the high scale image and low scale image of information
2. using resiudal network and deconvolution to combine them

## 5. Update model parameters with accumulated gradients  
Theoretically, when calculating gradients in back propagation, it always updates parameters by 1 training data.  
Therefore, in order to make program execution more efficient, we accumulate gradients and update at once.  
  
## 6. Google Inception-v3
write by myself

## Reference  
1. WU, Yuxin; HE, Kaiming. Group normalization. arXiv preprint arXiv:1803.08494, 2018.  
2. ZHANG, Han, et al. Self-attention generative adversarial networks. arXiv preprint arXiv:1805.08318, 2018.  
3. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
LIU, Rosanne, et al. An intriguing failing of convolutional neural networks and the coordconv solution. In: Advances in Neural Information Processing Systems. 2018. p. 9627-9638.  
4. LIN, Tsung-Yi, et al. Feature pyramid networks for object detection. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017. p. 2117-2125.  
5. accumulated gradients  
https://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients  
6. SZEGEDY, Christian, et al. Rethinking the inception architecture for computer vision. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. p. 2818-2826.  

