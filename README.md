# Audio Classification

Sound classification with PyTorch  
Tools :    
- torchaudio, torchvision, and librosa for sound effects
- UrbanSound8K Dataset [[1]](#1).  
- DenseNet classifier [[2]](#2).  
I've also trained a model on a little custom CNN classifier (with 4 convolutions layers)  
- Data processing and training inspired from Palanisamy, K. [[3]](#3).  

## Data augmentation
I've linked an IPython notebook [preprocess.ipynb](./howto/preprocess.ipynb) to explain how to load, process, and augment a sound.

## Sound to Image  
I've linked an IPython notebook [MelSpectrogram.ipynb](./howto/MelSpectrogram.ipynb) to explain how an image could be created from a sound.  
This way, it is possible to apply computer vision techniques to classify sounds.  
![spectrograms.png](./doc/spectrograms.png)


## References  
<a id="1">[1]</a> 
https://urbansounddataset.weebly.com/urbansound8k.html  
<a id="2">[2]</a> 
Huang, Gao, et al. "Densely connected convolutional networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.  
<a id="3">[3]</a> 
Palanisamy, K., Singhania, D., & Yao, A. (2020). Rethinking CNN models for audio classification. arXiv preprint arXiv:2007.11154  