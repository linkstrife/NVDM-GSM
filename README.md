# Neural Variational Topic/Document Model with Gaussian Softmax Construction tensorflow implementation
This is an unofficial tensorflow implementation of the neural variational topic model with Gaussian Softmax Construction (GSM). The original model is Neural Variational Document Model (NVDM). The code paragraph of BOW-VAE impelementation is adpted from https://github.com/ysmiao/nvdm. 

# Reference
[Discovering Discrete Latent Topics with Neural Variational Inference](http://proceedings.mlr.press/v70/miao17a.html). Miao et al., ICML 2017.
This implementation is able to reproduce the results presented in the paper.

# Build Version
tensorflow 1.14, on windows 10.

# Dataset
20news group. Incuding 2 .feat files and 1 .new vocabulary. This dataset is directly copied from https://github.com/ysmiao/nvdm/tree/master/data/20news and has not been modefied.

# Notice
Part of the code comments are written in Chinese, please feel free to translate by available tools.
One can redefine the FLAGS.model_type to switch from a topic model to a document model.

# Author
Lihui Lin, School of Computer and Data Science, Sun-Yat Sen University.
