# Accelerate LLM Inference with Few-Step Parallel Decoding

**TL;DR:** In this article, we introduce consistency large language models (CLLMs), a new family models developed with our proposed techniques to reduce inference latency using Jacobi decoding. 



## Background: Jacobi Decoding

Large language models (LLMs) are transforming the landscape of human lives, From programming to offering legal and health advice. However, during inference, LLMs generate responses token by token using auto-regressive (AR) decoding as shown in Figure 1, leading to high latency for longer responses.

<p align="center"><img src="clm_objective.png" alt="autoregressive" width="250"></p>
<p align="center">Figure 1: illustration of conventional AR decoding: one token is generated at a time.</p>

Jacobi decoding is first introduced by [this paper](https://arxiv.org/abs/2305.10427). 

<p align="center"><img src="jacobi_objective.png" alt="autoregressive" width="350"></p>
<p align="center">Figure 2: illustration of Jacobi decoding: n-token sequence is fed into the LLM and iterates until convergence.</p>

### Jacobi Trajectory

### Limitations of Vanilla Jacobi Decoding

## Consistency LLMs (CLLMs)

##  Experiments

### Fast Forwarding and Stationary Tokens

### Results


## Final words
We invite you to refer to the [our paper](TODO) for more details! Please stay tuned for code and CLLM checkpoint release!
