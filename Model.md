# Model (forward pass)



## Quick Understanding of Deep Learning

-   Teacher: Us
-   Student: Machine (Computer)

-   **Hints before learning:** 

    a Function with Parameters to be Learnt & some Hyper-parameters (MANUALLY PROVIDED) => $Building\ the\ Model$

-   **Data for learning:** 

    a Training Set => $Train\ Time$

-   **Notes summarizing the learning:** 

    the Function with learned Parameters => $Learnt\ Model$

    ==To come up with the right Function is the key stage of Deep Learning!!!==

-   **Data for checking learning effect:** 

    a Validation Set => $Validation\ Time$

-   **Consolidate the knowledge:** 

    the Training-Validation Loop 

-   **==Data for the FINAL EXAM:==** 

    ==a Testing Set unseen through any stages before => $Test\ Time$==



## Hyper-Parameters

-   **(Initial) Learning Rate (set in Optimizer)** 
-   **Momentum (set in Optimizer)**
-   **Dropout (set in NN Module)**
-   **Weight-decay (set in Optimizer)**



## Evaluation (of Generalization) & Solutions

在统计学和机器学习中，**bias（偏差）**和**variance（方差）**是两个重要的概念，它们是用来<u>衡量模型预测**误差的主要来源**</u>。在设计和训练机器学习模型时，存在一个bias-variance权衡的问题。如果**模型过于简单并且具有有限的参数（模型的特征太弱，不足以区分不同样本，对样本变化不敏感），则可能会有高偏差和低方差。在另一方面，如果模型过于复杂并且具有大量的参数（模型的特征太强，过于区分不同样本，对样本变化太敏感），则可能会有低偏差和高方差。**在这两种情况下，模型在测试集上的表现都可能不佳。因此，我们的目标是**找到偏差和方差之间的最佳平衡点，以实现最佳的泛化能力**。

1.  **Bias（偏差）**：偏差是指模型的预测值与实际值的平均差异，或者说是模型的预期预测与真实预测之间的差距。简单来说，高偏差模型会忽视数据中的细节，这通常会导致欠拟合（underfitting），即模型在训练集和测试集上的表现都不好。

    <img src="images/image-20230724173953912.png" alt="image-20230724173953912" style="zoom:50%;" />

    *($\mu$的一阶矩是无偏估计)*  

2.  **Variance（方差）**：方差是指模型预测的变化性或离散程度，即同样大小的不同训练集训练出的模型的预测结果的变化。如果模型对训练数据的小变动非常敏感，那么模型就有高方差，这通常会导致过拟合（overfitting），即模型在训练集上表现很好，但在测试集（即未见过的数据）上表现差。

    <img src="images/image-20230724174527318.png" alt="image-20230724174527318" style="zoom:50%;" />

    *（$\sigma$的一阶矩是有偏估计）*

    找同一模型公式的最佳参数相当于对着靶子开一枪，**偏差**相当于瞄准点偏离了实际靶心，**方差**相当于射击点偏离了瞄准点：

    <img src="images/image-20230724174659994.png" alt="image-20230724174659994" style="zoom:50%;" />

    <img src="images/image-20230724175124356.png" alt="image-20230724175124356" style="zoom:50%;" />

    （不同的 $f$ 对应用不同数据点训练出的同一模型公式的不同模型参数）

    不同公式的模型的训练结果分布图：

    <img src="images/image-20230724175523663.png" alt="image-20230724175523663" style="zoom:50%;" />

    越复杂的模型受数据集变化的影响越大，变动幅度越大，不同 “universe” 的训练结果间差距更大，即方差更大：

    <img src="images/image-20230724175711069.png" alt="image-20230724175711069" style="zoom:50%;" />

    但方差大时覆盖的范围也越大，取均值（一阶矩）时更容易覆盖“靶心”，因此越复杂的模型偏差越小：

    <img src="images/image-20230724180100173.png" alt="image-20230724180100173" style="zoom:50%;" />

    **结论：**复杂模型（参数多）最理想优化结果和真实情况间 Loss （即Bias）更小，但实际的 training 结果和理想情况差距很大，因为参数多意味着函数空间更大更难优化，需要更大的 Data Set。

<img src="images/image-20230724180242714.png" alt="image-20230724180242714" style="zoom: 33%;" />

<img src="images/image-20230724180355587.png" alt="image-20230724180355587" style="zoom: 33%;" />

<img src="images/image-20230724180533868.png" alt="image-20230724180533868" style="zoom: 33%;" />

<img src="images/image-20230724180635703.png" alt="image-20230724180635703" style="zoom:50%;" />

Should：（把所有训练集划分成训练集、验证集）① 用训练集训练 【取训练集对应的最佳参数】-> ② 用验证集再训练 【取①以后验证集对应的最佳参数】-> ③ 用所有数据集最后再训练【取②以后所有数据集对应的最佳参数】

<img src="images/image-20230724180852393.png" alt="image-20230724180852393" style="zoom:50%;" />

<img src="images/image-20230724181425216.png" alt="image-20230724181425216" style="zoom:50%;" />

**虽然直接优化目标是Loss小，而非偏差、方差是否最佳平衡（这也很难作为优化目标），**<u>但一般训练集Loss足够小且测试集Loss不严重高于训练集，就是比较平衡的结果</u>。

**解决模型复杂度（参数多导致的巨大函数空间）和与真实值的偏差之间的矛盾**：<u>用多层级的神经网络 =》**深度学习（见 Why Deep?????????）**</u>

<img src="images/image-20230724234749795.png" alt="image-20230724234749795" style="zoom: 33%;" />



## Deal With Over-fitting

***预测结果不好（准确度/ 损失值）但训练结果更好 —— 过拟合***

### Early Stopping

准备验证集（一般是所有已知数据**训练:验证=8:2**），一边训练以更新参数一边在每个epoch结束用验证集求误差（训练误差也求出来，用于比较观察是否过拟合）：

-   以验证集误差小于某阈值或更新多少次后不再出现更小值为 early stopping 标志；
-   以验证集误差在多少次后不再小于等于训练集误差为标志。

期间保留<u>使验证集误差最小</u>的参数直到最后。

### Dropout

类似决策树的剪枝，但 Dropout 一般是随机舍去神经元，剪枝是经泛化性能的比较后减去决策分支。

### Shuffle

每个epoch开始前对batches洗牌（在 `DataLoader` 处设置，训练集和验证集开启，测试集关闭）

### Maximize the Margin + Soft Margin

在分类问题中，最大化有训练集学得的划分超平面两侧的间隔（参考《机器学习》支持向量机），并且允许部分训练数据出错，即落入间隔内而不在硬间隔外部的严格分类区域（软化间隔）。

### Support Vector Regression (SVR)

在回归问题中，容忍落在间隔带内的出错数据，不计算其误差到 Loss ，即给 Loss Function 添加一个不敏感损失函数项，类似正则化（参考《机器学习》支持向量机）。

### L2 Regularization (Weight-decay)

See [Loss Function](D:\CAMPUS\AI\MachineLearning\LossFunction.md)

***正则项本质上是支持向量机的最大化间隔法产生的，参考《机器学习》p122,123,133*** 

### (In Classification) Weaken Features of a Each Class

估计各类别的分布时（建模时），**使它们拥有相同的某参数**，该参数<u>由各类别分别用对应训练集估计出的参数再对样本数**加权平均**得到</u>

### Simplify Network Structure to Cut Down Sensitivity of  Features' Alternation

See **<u>CNN</u>**

### Batch Normalization

See  [DataProcessing](D:\CAMPUS\AI\MachineLearning\ML_MDnotes\DataProcessing.md)---**<u>Batch Normalization</u>**

### Cross Validation, K-Fold



## Coding Tips

-   **`torch.nn`** 内置各种NN模块，可作为完整模型架构或隐藏层，也可自定义**`torch.nn.Module`** 的派生类作为模型类，里面使用内置NN模块建立模型架构（用**`torch.nn.sequence`**）

-   **`torch.nn.Module`** 的可用方法（含必须实现的）见 **[Method of Class "Module"](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)**

-   **`torch.nn`**内置的NN模块（Layers）见 [**Build-in NN Model (Layers Block)**](https://pytorch.org/docs/stable/nn.html)

-   Training框架：

    ```python
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 正向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数
    ```



## Stable Training

```python
# Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
```

`nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)`这一行代码用于梯度裁剪，其目的是为了防止在神经网络训练过程中出现"梯度爆炸"的问题。

梯度爆炸是指在训练过程中，模型的参数的梯度变得非常大，以至于更新步长过大，导致模型无法收敛，甚至结果溢出。当计算出的梯度大于某个设定的最大值（在这里是10）时，这个函数将会按比例缩小梯度，使得其不超过这个最大值，从而避免梯度爆炸。

`grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)`这行代码会返回所有模型参数梯度的L2范数。

梯度裁剪是很常见的在训练深度学习模型，特别是在训练深度神经网络和循环神经网络（RNN）时，用于保持模型的稳定性。



## Ensemble

**"Ensemble"（集成）**是一种机器学习策略，它结合了多个模型的预测，以产生一个最终的预测结果。这种方法的主要目标是通过组合多个模型来减小单一模型的预测误差。

主要的集成方法有：

1.  Bagging（Bootstrap Aggregating）：通过从训练数据中有放回地随机抽样来生成不同的训练集，然后在每个训练集上训练一个模型。最后，所有模型的预测结果通过投票（分类问题）或平均（回归问题）来产生最终预测。随机森林就是一个典型的Bagging的例子。
2.  Boosting：一种迭代的策略，其中新模型的训练依赖于之前模型的性能。每个新模型都试图修正之前模型的错误。最著名的Boosting算法有AdaBoost和梯度提升。
3.  Stacking：使用多个模型来训练数据，然后再用一个新的模型（叫做元模型或者二级模型）来预测这些模型的预测结果。

在许多机器学习竞赛中，集成学习被证明是一种非常有效的方法，因为它可以**减少过拟合，增加模型的泛化能力，从而提高预测性能**。



## Why Deep?????????

-   在比浅层模型的各层参数个数少的情况下，通过增加深度来增加模型复杂度以减小偏差，且最终整体参数个数也少于浅层但各层参数更多的模型【复杂化过程对参数的利用率更高】，这样训练需要的数据也更少（在真实情况复杂又有潜在规律\<比如语音、图像的模型>时效果更突出！！！）
-   利用多层激活函数逐层引入非线性逼近复杂的函数



## Classic Model Structures

### Linear Regression

```python
class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()   # must call this explicitly!!!!!
        # structure of the LR
        self.layers = nn.Linear(input_dim, output_dim)

    def forward(self, x): # only need to customize forward()
        return self.layers(x)
```

****

### Basic fully connected NN (FCN/ multi-layer feedforward NN) 

```python
class My_Model(nn.Module):  # derived from build-in class "nn.Model"
    def __init__(self, input_dim):
        super().__init__() 
        # structure of the DNN
        self.layers = nn.Sequential(  
            nn.Linear(input_dim, <s1>),
            nn.ReLU(), # activator
            nn.Linear(<s1>, <s2>),
            nn.ReLU(),
            nn.Linear(<s2>, <s3>),
            nn.ReLU(),
            <...>
            nn.Linear(<sn>, output_dim)
        )

    def forward(self, x):  
        x = self.layers(x)  # input a batch of Features (x) to the DNN (in-place)
       #  x = x.squeeze(1)  # (B, 1) -> (B) [simplify the shape of tensor]
        return x
```

****

### CNN ( Convolutional NN)

-   ***从网络内部结构改变模型性能***
-   图像数据（矩阵）常用
-   需要添加 **Spatial Transformer Layers** 
-   据具体问题考虑是否使用 **Pooling**（不是所有数据都能用！！一般图像可以）

**简化网络使得非全连接，以减少网络层权重（参数）数，以削弱网络的特征敏感度，防止过拟合**

**`torch.nn.Conv2d`** 是 PyTorch 提供的二维卷积操作。卷积在图像处理和计算机视觉中非常重要，特别是在卷积神经网络中。

**参数**：

-   `in_channels`: 输入数据的通道数。例如，对于彩色图像，`in_channels` 为3（分别对应红、绿、蓝三种颜色）；对于灰度图像，`in_channels` 为1。
-   `out_channels`: 卷积产生的特征图（Feature Maps）的数量。这个参数可以被视为学习特征的数量。
-   `kernel_size`: 卷积核的尺寸，可以是单个整数或一个包含两个整数的元组（分别表示高和宽）。卷积核是用于扫描图像的滑动窗口。
-   `stride`: 卷积核移动的步长，可以是单个整数或一个包含两个整数的元组。步长控制了卷积核在输入图像上滑动的速度。
-   `padding`: 在输入数据的周围添加的零的层数。可以是单个整数或一个包含两个整数的元组。补零可以保证输出特征图的尺寸不变或者控制输出特征图的尺寸。
-   `dilation`: 卷积核元素之间的间距。它可以被用于控制卷积核覆盖的空间尺寸，而不改变卷积核中的元素数量。
-   `groups`: 控制输入和输出之间的连接。默认情况下，groups=1，意味着每个输入通道与每个输出通道都连接。如果groups=2，那么前一半的输入通道与前一半的输出通道连接，后一半同理。当`groups=in_channels`和`out_channels`时，卷积操作就变成了深度卷积。
-   `bias`: 如果设置为True，那么向卷积添加偏置。默认为True。

```python
class My_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input_size: [3, 128, 128]
        self.cnn_pool = nn.Sequential(  # CNN
            nn.Conv2d(3, 64, 3, 1, 1),  # output_size: [64, 128, 128]
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # Pooling to: [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # output_size: [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # Pooling to: [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # output_size: [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # Pooling to: [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # output_size: [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # Pooling to: [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # output_size: [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # Pooling to: [512, 4, 4]
        ) 
        self.fullc = nn.Sequential(  # Fully-connected NN
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn_pool(x)
        out = out.view(out.size()[0], -1)  # 展平操作（Flatten）：常常在卷积层和全连接层之间进行，因为全连接层需要的输入是一维的
        return self.fullc(out)
```

****

### RNN (Recurrent Neural Network)

**==已经可以被Self-attention取代！！！见 <u>Self-attention</u>==**

**[PARAMETERS of nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?highlight=rnn#torch.nn.RNN)**  

**循环神经网络（Recurrent Neural Network，RNN）**是一种专门处理**序列数据**（例如，时间序列数据\<音频>或文本）的神经网络（可以对语音分类，不用一般的分类模型）。RNN与普通的全连接神经网络和卷积神经网络不同，它能够**处理序列长度可变的数据**，<u>在处理每个元素时，它都会记住前面元素的信息</u>。这就是它被称为“循环”的原因。

RNN的基本思想是在神经网络的**隐藏层之间建立循环连接**。**每一步都会有两个输入：当前步的输入数据和上一步的隐藏状态（于是每次model会有两个输出值，训练过程注意左值设为`output, _`）。**然后，这两个输入会被送入网络（通常是一个全连接层或者一些更复杂的结构，如LSTM或GRU单元），然后产生一个输出和新的隐藏状态。这个新的隐藏状态将被用于下一步的计算。

这个过程可以写作如下形式的数学公式：
$$
h_t = f(h_{t-1}, x_t)
$$
其中，$h_t$是在时间t的隐藏状态，$x_t$是在时间t的输入，$f$是一个非线性函数，它定义了如何从前一步的隐藏状态和当前的输入计算当前的隐藏状态。

RNNs在许多不同的任务中都被证明是非常有用的，特别是在处理语音识别、语言建模、机器翻译等涉及序列到序列的转换的任务中。然而，它们也有一些已知的问题，特别是在处理长序列时，它们往往会遇到所谓的梯度消失和梯度爆炸问题。这些问题已经有一些解决方案，例如**长短期记忆（Long Short-Term Memory，LSTM）网络**和**门控循环单元（Gated Recurrent Unit，GRU）**。

```python
class My_Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = nn.RNN(input_size, hidden_size, batch_first=True) 
        # or "nn.LSTM"/ "nn.GRU"

    def forward(self, x):  # 2 outputs !!!!!
        # x => (batch_size, sequence_length, input_size)
        # hidden => (num_layers, batch_size, hidden_size)
        x, hidden = self.layers(x)
        # (LSTM: x, (hidden, cell_state) = ...)
        return x, hidden
```

如果`batch_first=True`，那么输入和输出张量的形状应该为`(batch, seq, feature)`，即批量大小（batch size）是张量的第一维度。如果`batch_first=False`（这是默认设置），那么输入和输出张量的形状应该为`(seq, batch, feature)`，即序列长度（sequence length）是张量的第一维度。在大多数情况下，`batch_first=True`可能更直观，因为我们通常会将批量大小作为第一维度。但有时，出于性能优化或与特定库的兼容性考虑，可能会需要`batch_first=False`。

****

### Self-attention

***CNN 是 Self-attention 的特例！！！！！！！！！！***

**Self-attention是transformer的特例！！！！！！！（input序列的元素各与一个output序列元素对应）**

自注意力机制（Self-Attention）原本在自然语言处理领域非常受欢迎，但也已经开始在计算机视觉领域发挥作用，具体的应用形式可能有所不同，但核心思想都是对输入的各个部分赋予不同的注意力权重。

试想我们正在观看一张图片，我们的视线并不是平均分布在整个图片上，而是会集中在一些我们认为更重要的地方，比如人物的脸部，或者是图片中的主要物体。这种行为实际上是一种自然而然的注意力机制，我们关注某些部分多于其他部分。自注意力机制就是试图模拟这种行为，让计算机模型也能“聚焦”于输入的关键部分。

在计算机视觉中，自注意力可以用于捕捉图像中长距离的依赖关系。比如在一张人物照片中，虽然眼睛和嘴巴在空间位置上可能相距较远，但它们之间存在很强的语义关联。传统的卷积神经网络（CNN）由于其局部感受野的特性，可能难以捕捉这种长距离的关系，而自注意力机制能有效解决这个问题。

一个具体的例子是"Vision Transformer"（ViT）模型，这个模型是基于Transformer（一种主要使用自注意力机制的模型结构）设计的，它将图像划分为一系列的小块，然后像处理文本那样处理这些小块。通过这种方式，ViT可以把注意力放在图像中最重要的部分，从而提高模型的预测性能。

总的来说，自注意力机制在计算机视觉中的应用是一种很有前景的研究方向，它能帮助模型关注到图像中最重要的部分，从而提高模型的性能。

**Self-attention下游应用：https://www.youtube.com/watch?v=yHoAq1IT_og （包括Transformer）**

****

### CNN & RNN

==**CNN常用作图像分类，RNN常用作文本生成，两者结合可以做 Image Captioning：**==

![image-20230805110434586](images/image-20230805110434586.png)

****

### Cross-attention

<img src="images/image-20230726132533226.png" alt="image-20230726132533226" style="zoom:50%;" />

<img src="images/image-20230726132832992.png" alt="image-20230726132832992" style="zoom:50%;" />

****

### Transformer

是 **Sequence 2 Sequence Model** 的一种！！！

**输入任意长序列，输出由机器学得长度的序列。** 

Transformer是一种在深度学习和自然语言处理（NLP）中广泛使用的模型架构。Transformer模型最初在"Attention is All You Need"这篇论文中被提出，并已成为许多现代自然语言处理模型的基础，比如BERT、GPT-2/3、T5等。

以下是Transformer架构的一些关键特性：

1.  **自注意力机制（Self-Attention Mechanism）**：这是Transformer最重要的组成部分之一，也被称为Scaled Dot-Product Attention。自注意力机制使模型能够处理输入序列中的每个元素，并确定其与序列中其他元素的关系。
2.  **位置编码（Positional Encoding）**：由于Transformer模型本身没有任何关于元素顺序（即在序列中的位置）的信息，因此我们需要添加位置编码以保留这些信息。
3.  **多头注意力（Multi-Head Attention）**：在实践中，我们通常会使用多头注意力，它包含了多个并行的自注意力层。这可以让模型同时关注输入序列中的多个不同位置，从而捕获更丰富的信息。
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

****

### Residual

常和 **Self-attention** 一起用

****

### Generative

***（如何假设分布：多元特征样本则对训练集每类样本都分别设为一个高斯分布<可共用方差并共同计算似然函数>，无论是否是二元分类；单元特征样本且二元分布则直接对所有训练集设为一整个伯努利分布）***

****

### Logistic Regression

***（不需要自己假设分布！！！）***

和 Linear Regression 的区别仅在<u>**最终**输出时**多加一层 Softmax**</u>（这是 Softmax 作为生成概率分布的用途时的用法，<u>若是别的用途则在隐藏层也会加</u>！！！！sigmoid是特殊的sigmoid，二分类时直接用sigmoid；**Logit \<对数几率，log odds>** 狭义即指 sigmoid 的反函数，广义指<u>未经 Softmax 的输出</u>）

```python
class My_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out
```

**若后面 Loss 用 CrossEntropy 则 Model 就直接用 Linear Regression ！！！因为 CrossEntropy 已包含 Softmax！！！（这就是为什么pytorch没有专门的 Logistic Regression 函数）**

>   **Softmax** 的用途：
>
>   -   在网络末端由线性函数的输出生成概率分布（映射到 [0,1]）
>   -   在网络末端由线性函数的输出生成概率分布之前，对<u>每个特征</u>分别实现特征变换使得线性可分（Logistic Regression 网络中用）【也可加多层变换】
>   -   在网络内部每层输出后用作激活函数引入非线性使得模型可以更好地逼近复杂的函数（但一般用ReLU）

****

### GNN (Graph Neural Networks)

图神经网络（Graph Neural Networks，简称GNN）是一种强大的用于处理图形数据的深度学习架构。它们已经在各种领域（包括社交网络分析、推荐系统、生物信息学等）显示出优异的性能。在计算机视觉领域，GNN也找到了一些有趣且有用的应用。

虽然一般的计算机视觉任务（例如图像分类或目标检测）通常可以直接使用卷积神经网络（CNN）来处理，但是一些更复杂的问题（例如对场景理解的需求，包括对象的关系或上下文信息）可能会受益于GNN。例如，一张图片可以被看作是一个图，其中节点表示物体，边表示物体间的关系，GNN就可以用来处理这样的问题。

此外，GNN在视频理解和3D数据处理中也有很多应用，因为这些数据形式的时间和空间关系可以自然地被建模为图。例如，在视频分析中，GNN可以帮助捕捉不同帧之间的关系；在3D点云分析中，GNN可以更好地捕捉空间点之间的复杂关系。

总的来说，GNN为计算机视觉提供了一种强大的工具，尤其是在需要理解对象间复杂关系和上下文信息的场景中。然而，它并非适合所有的视觉任务，而是作为一个补充工具，和CNN等其他视觉处理模型共同使用。

****

### Pointer

**特点：使用了 Copy Mechanism（部分输出的信息直接复制自输入） **

Pointer Network是一种序列到序列（Seq2Seq）模型，它的出现是为了解决Seq2Seq模型在处理某些特定问题时的困难，如排序问题，最短路径问题等。这些问题的一个共同特点是，**输出的元素来自于输入序列的元素**，而不是从一个固定的词汇表中选取。

传统的Seq2Seq模型会在每一步输出一个来自固定词汇表的词，而Pointer Network则会在每一步输出输入序列中的一个位置。这是通过使用注意力机制并将其输出解释为一个概率分布来实现的，这个分布表示生成每个输入元素的概率。在生成序列的每一步，模型都会"指向"输入序列中的一个元素，因此得名Pointer Network。

Pointer Network对于那些输出是输入的一个排列或者子序列的问题特别有用。例如，在车辆路径问题（Vehicle Routing Problem）中，输入是一组城市的坐标，输出是访问所有城市的最短路径，这就是输入的一个排列。在这种情况下，使用Pointer Network比使用传统的Seq2Seq模型更有效。

****

### Generator

#### GAN（生成式对抗模型）

https://youtu.be/4OWp0wDu6Xw

https://youtu.be/jNY1WBb8l4U

https://youtu.be/MP0BnVH2yOo

https://youtu.be/wulqhgnDr7E



## 2 Modes of Model (nn.Module) & ` with torch.no_grad()`:

****

|                                                              |     训练时`.train()`      |            验证/ 测试时`.eval()`             |
| :----------------------------------------------------------: | :-----------------------: | :------------------------------------------: |
| 添加**`with torch.no_grad():`**禁止在`.forward()`时提前做好求梯度的预备计算并存储（`.forward()`默认内置这一操作） |             ❌             |       ✔️（节省存储且加速`.forward()`）        |
|                  自动调整学习率（如果可以）                  |             ✔️             |                      ❌                       |
|         据概率（给定参数）随机关闭神经元（dropout）          |             ✔️             |                      ❌                       |
|              `.BatchNorm2d`（如果模型中设置了）              | 计算每个batch的均值和方差 | 使用在训练过程中计算得到的移动平均均值和方差 |
