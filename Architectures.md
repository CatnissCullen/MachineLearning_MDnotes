# Architectures

****



## Encoder-Decoder

-   **First Conv: **in spite of any input channels, first Conv to a certain number of channels (usually 32 or 64)  
-   **Throughout (after initial Conv and before final Reverse Conv): **
    -   CNN Module for each encoder layer --- **RELU-CONV-NORM + maybe some RES**
    -   Reverse CNN Module for each decoder layer --- **RELU-CONV^T-NORM-DROPOUT + maybe some RES**
-   **Last Conv:** Conv to output channels and centralize the pixel values --- **CONV-TANH** 
-   ***Encoder: Down-sample (Pool using CNN) the any-length input seq. into a fixed-length seq., to filter out the superfluous info. and pool the essential info. for final output.***  
-   ***Decoder: Up-sample the fixed-length seq. into the final output of an any-length seq. using other conditions.***
-   **Complete structure: ** input image -> n-layer Encoder -> latent -> n-layer Decoder -> output image  

### Auto Encoder

一般的 Encoder-Decoder 用于有监督学习，目标输出是输入的某种转化表示（如图像转声音、声音转文本），有目标输出的真实样本，**在目标输出与实际输出间计算损失值**；而 Auto Encoder 是无监督的，它要求输出内容本身（如图像与图像相似、声音与声音相似）尽可能接近输入，也可人为添加新的特征条件（conditional，这不叫监督，有真实目标的才叫监督，supervised），没有目标输出的真实样本，**在目标输出的所属分布与输入的所属分布间计算损失值**，是特殊的 Encoder-Decoder，latent 的长宽通常为1



## U-net

类似网络模型，在 Encoder-Decoder 基础上，同层的 Encoder 输入在 channel 维度上**拼接**（和ResBlock的区别！）到 Decoder输出（skip-connection），再输入到下一层 Decoder

**⭐ 解决有用的信息在 Encoder 卷积过程过度丢失的问题**



## Transformer

是 **Sequence 2 Sequence Model** 的一种！！！

**输入任意长序列，输出由机器学得长度的序列。** 

Transformer是一种在深度学习和自然语言处理（NLP）中广泛使用的模型架构。Transformer模型最初在"Attention is All You Need"这篇论文中被提出，并已成为许多现代自然语言处理模型的基础，比如BERT、GPT-2/3、T5等。

以下是Transformer架构的一些关键特性：

1.  **自注意力机制（Self-Attention Mechanism）**：这是Transformer最重要的组成部分之一，也被称为Scaled Dot-Product Attention。自注意力机制使模型能够处理输入序列中的每个元素，并确定其与序列中其他元素的关系。

2.  **位置编码（Positional Encoding）**：由于Transformer模型本身没有任何关于元素顺序（即在序列中的位置）的信息，因此我们需要添加位置编码以保留这些信息。

    https://zhuanlan.zhihu.com/p/106644634

    ​	While transformers are able to easily attend to any part of their input, the attention mechanism has no concept of token order. However, for many tasks (especially natural language processing), relative token order is very important. To recover this, the authors add a positional encoding to the embeddings of individual word tokens.

    Let us define a matrix $P \in \mathbb{R}^{l\times d}$, where $P_{ij} = $ 
    $$
    \begin{cases}
    \text{sin}\left(i \cdot 10000^{-\frac{j}{d}}\right) & \text{if j is even} \\
    \text{cos}\left(i \cdot 10000^{-\frac{(j-1)}{d}}\right) & \text{otherwise} \\
    \end{cases}
    $$
    The positional encoding is created using sine and cosine functions of different frequencies. The sine is applied to even indices in the positional encoding array, and cosine to odd indices.

    Rather than directly passing an input $X \in \mathbb{R}^{l\times d}$ to our network, we instead pass $X + P$.

3.  **多头注意力（Multi-Head Attention）**：在实践中，我们通常会使用多头注意力，它包含了多个并行的自注意力层。这可以让模型同时关注输入序列中的多个不同位置，从而捕获更丰富的信息。

    ​	***Steps:***

    1.  **Linear Layers**: First, you pass your `query`, `key`, and `value` through separate linear layers.
    2.  **Splitting Heads**: Then you'll split these into multiple heads. In PyTorch, this is often done by reshaping.
    3.  **Scaled Dot-Product Attention**: For each head, perform the scaled dot-product attention.
    4.  **Concat Heads**: Concatenate the heads back together.
    5.  **Final Linear Layer**: Pass through one more linear layer.

4.  **前馈神经网络（Feed Forward Neural Networks）**：每个Transformer层除了注意力子层外，还有一个前馈神经网络，它在每个位置独立地应用于注意力子层的输出。

5.  **残差连接（Residual Connections）和层归一化（Layer Normalization）**：Transformer模型中使用了残差连接和层归一化技术，这些技术有助于训练更深的模型。

训练技巧：

1.  **Copy Mechanism**
2.  **Guided Attention**
3.  **Beam Search**

Transformer模型由于其强大的性能和灵活性，不仅在NLP中得到广泛应用，还开始在图像处理等其他领域中找到应用。

近几年Transformer结构在计算机视觉领域的应用也越来越广泛。传统上，卷积神经网络（Convolutional Neural Networks, CNN）一直是计算机视觉任务，如图像分类、物体检测和语义分割等的主流模型。然而，Transformer架构因其在处理序列数据方面的优势，开始被广泛应用于计算机视觉任务。

例如，Vision Transformer (ViT) 是一个将 Transformer 架构应用于图像分类的模型，它把图像切分为一系列小的图像块，然后把这些图像块序列化后输入到 Transformer 中。这种方法使得模型能够更好地处理图像中的长距离依赖关系，并且可以更好地泛化到更大的图像。

另一方面，DETR (DEtection TRansformer) 将 Transformer 用于物体检测任务，消除了传统物体检测算法中的许多手工设计的部分，如锚框和非最大抑制等。这使得物体检测更加直接，也使得模型更容易训练。

因此，尽管最初 Transformer 是为了解决自然语言处理任务而设计的，但其在计算机视觉中也展现出了强大的潜力，已经在许多领域得到了应用。