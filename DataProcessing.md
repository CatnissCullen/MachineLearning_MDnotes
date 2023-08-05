# Data Processing

****



## Basics Loading Formats

### Image

-   在输入模型前需要设置 Spatial Transformer Layer 学习如何将图片大小和方向统一（see **<u>Spatial Transformer Layer</u>**）
-   看作 3 维矩阵（RGB Channels、长度、宽度）输入CNN，输入全连接网络前拉直（flatten）成一维向量。

（具体见卷积神经网络 goodnotes）

### Audio

看作向量，将音频段划分成有重叠的帧窗口，在每个帧窗口上计算处特征值（如MFCC）作为向量的一个元。

以下是计算MFCC（Mel-Frequency Cepstral Coefficients）的具体步骤：

1.  **预加重（Pre-emphasis）**：对原始音频信号应用预加重滤波器，它会增加信号的高频部分的幅度，以补偿人耳对高频声音的敏感性较低的问题。这通常通过将信号通过一个差分滤波器来实现。
2.  **帧分割（Frame Stripping）**：由于语音信号是非平稳的，我们通常假设在短的时间帧内，语音信号是平稳的。因此，我们将整个音频信号分割成20-30ms的帧，每个帧之间有一定的重叠，通常是10-15ms。
3.  **窗函数（Windowing）**：在每个帧上应用窗函数（如汉明窗或汉宁窗），以最小化帧边缘的信号不连续性。这是为了防止在做傅立叶变换时，由于边缘不连续造成的频谱泄漏。
4.  **快速傅里叶变换（FFT, Fast Fourier Transform）**：对窗口化的帧进行快速傅里叶变换，得到每个帧的频谱，然后计算功率谱。
5.  **Mel滤波器组（Mel Filter Bank）**：应用Mel滤波器组到功率谱上。Mel滤波器组是一组在Mel频率上均匀分布的三角滤波器，它模拟了人类听觉系统对不同频率的不同感知能力。然后，对每个滤波器的输出进行求和，得到Mel频率上的功率谱。
6.  **对数取值（Logarithm）**：对Mel功率谱取对数，模拟人耳对声强的感知也是对数的。
7.  **离散余弦变换（DCT, Discrete Cosine Transform）**：对对数Mel功率谱进行离散余弦变换，得到Mel-Frequency Cepstral Coefficients。这个步骤将频谱的高度相关性降低。

在整个过程中，我们可以根据具体的应用需求，选择提取不同数量的MFCCs。

可能并不适合以单个帧窗口为样本，于是可以用前后的多个帧窗口辅助对单个帧窗口的预测，称为**“污染”（Contamination）**。



## Kernel Trick/ Kernelize

用核函数实现不可分训练集的高维线性可分：

>   只要一个**对称函数**对应的**核矩阵半正定**，该**对称函数**<u>就能作为**核函数**使用</u>，因为该**半正定矩阵**总能找到一个与之对应的**映射**使得样本点集能从原始空间<u>映射到更高维空间使得其线性可分</u>，即任何**核函数**都隐式定义了一个称为RKHS（$Reproducing Kernel Hilbert Space$）的特征空间。
>
>   ==***文本数据通常用线性核，情况不明时可先尝试高斯核。***==
>
>   *（参考《机器学习》支持向量机）*

核函数的解：

表示定理（见《机器学习》p137）是核方法的一个重要理论支持，它告诉我们，在一定的条件下，学习算法的解可以表示为所有训练样本的线性组合。这一点在支持向量机等基于核方法的算法中尤为重要，因为在这些算法中，只有支持向量（即在决定决策边界的样本）的系数不为零，其他样本的系数都为零，所以学习的模型实际上是由支持向量线性组合而成的。这就是为什么这些算法被称为"支持向量"机的原因。

换句话说，表示定理实际上告诉我们，在高维特征空间中找到的解，可以用原始空间中的数据通过核函数计算得到，从而避免了显式地在高维空间中进行计算，大大降低了计算复杂度。

线性判别分析（LDA）是一种常用的线性分类方法。它的基本思想是找到一个线性的投影方向，使得同类数据的投影点尽可能地接近，而不同类数据的投影点尽可能地远离。当数据是线性可分的时候，LDA可以很好地工作。但是，当数据是线性不可分的时候，LDA就会遇到困难。

这个时候，就可以引入核化的思想，通过引入核函数将原始的输入空间映射到一个新的特征空间中，然后在新的特征空间中应用LDA。这就是所谓的核线性判别分析（Kernel Linear Discriminant Analysis，简称KLDA）。



## Feature Engineering

*参考《机器学习》决策树*

**指标：**

***好的分类指标应该在分类后显著提高数据集有序性***

-   信息增益（分类前后**信息熵**之差，分类后用的是加权信息熵）
-   增益率（分类前后按特征可取值数量进行惩罚的信息增益率）
-   基尼指数（**基尼值**为数据集中任意两样本类别不一致的概率，其相反数即数据集纯度；基尼指数为以一个特征\<属性>划分数据集后数据集的加权基尼值）



## Data Augmentation

**Data Augmentation（数据增强）**是一种在训练深度学习模型时常用的技术，**通过创建修改版的现有数据来增加训练集的大小**。这种技术可以帮助<u>提高模型的泛化能力，防止过拟合</u>。

### IMAGE AUGMENTATION

在图像分类任务中，数据增强可能包括：

1.  图像翻转：水平翻转或垂直翻转图像。
2.  图像旋转：按一定角度旋转图像。
3.  缩放：增大或缩小图像的尺寸。
4.  裁剪：对图像进行随机裁剪。
5.  亮度、对比度、饱和度和色调调整：改变图像的这些属性。
6.  随机噪声：向图像添加随机噪声。

都可用 `torchvison.transforms` 实现（见 <u>**`torchvison.transforms`**</u>）

### TEXT AUGMENTATION

在自然语言处理（NLP）中，数据增强可能包括：

1.  同义词替换：将句子中的某些单词替换为其同义词。
2.  随机插入：在句子中随机插入一些新的单词。
3.  随机交换：随机交换句子中的两个单词。
4.  随机删除：随机删除句子中的一些单词。

总的来说，数据增强是一种有效的方式，可以通过**创建更多的训练样本**，使模型能够从更多的角度学习数据的特性。



## Processing Layers within the Model

### Spatial Transformer Layer ( for images)

***使得输入CNN的图片数据等大同向**（CNN不能兼容不等大同向的数据）*

让模型自主学习如何对输入的图片（此处用向量 $\vec x$ 表示）进行如下线性变换：
$$
\vec x' = A\cdot\vec x+\vec b
$$
其中，$A$ 是一个表示线性变换的方阵，**需要学习的参数个数即共 $n×n+n=(n+1)n$ 个**。

该种 Layer <u>**加入总模型中一起进行 GD 优化（直接视为模型的一部分）**，且可插入模型任意所需位置</u>。

### Feature Normalization Layer

用于增加模型泛化性能（平滑化 Loss 曲面不必要的、阻碍优化的隆起高地，因为 normalization 把属性值都归一化了，变小了）

#### Batch Normalization

`torch.nn.BatchNorm2d`是PyTorch中的二维批量归一化（Batch Normalization）层。这个层的主要作用是对每个mini-batch的输入数据进行归一化处理，以加速深度网络模型的训练并且提高模型的泛化能力。

通常被用在卷积神经网络（CNN）和全连接网络（FCN）的训练中。

**作用**：

1.  **深度神经网络训练**：在深度神经网络中，每一层的输入数据分布都会因为上一层参数的更新而改变，这种现象被称为“内部协变量转移”（Internal Covariate Shift）。这会导致网络需要更长的时间去适应这种变化，从而减慢训练速度。通过使用批量归一化，可以将每一层的输入数据强制转化为均值为0，方差为1的标准正态分布，减少内部协变量转移的影响，加快训练速度。
2.  **解决梯度消失和梯度爆炸问题**：在深度神经网络中，当数据的分布不均匀时，容易导致梯度消失和梯度爆炸的问题。通过批量归一化，可以调整数据的分布，缓解这一问题。
3.  **减少过拟合**：批量归一化在一定程度上具有正则化的效果，可以减少模型的过拟合现象。这是因为在训练过程中，每个mini-batch的数据都有自己的均值和方差，这增加了模型的噪声，相当于添加了一种形式的正则化。
4.  **允许使用更高的学习率**：批量归一化有助于网络的收敛，并且允许我们使用更大的学习率，这可以进一步加快训练速度。

可以插入到网络的任何部分，但**通常放在非线性变换（如ReLU激活函数）之前**。

具体参数含义如下：

1.  `num_features`：输入数据的通道数量。对于二维卷积层的输出数据，这通常就是卷积层的输出通道数。
2.  `eps`：这是一个用于保证数值稳定性的小常数，防止分母为零。默认值是1e-05。
3.  `momentum`：这是用于计算运行中的平均值和方差的动量。默认值是0.1。
4.  `affine`：布尔值，当设为True时，会为此层添加可学习的仿射变换参数。这意味着此层将有可学习的缩放参数（weight）和偏移参数（bias）。默认值是True。
5.  `track_running_stats`：当设为True时，该层将会保持运行中的平均值和方差的统计数据，用于在测试阶段进行批量归一化。当False时，只在每个mini-batch上计算统计数据，这在样本量较少的情况下可能会导致不准确。默认值是True。

例：

```python
pythonCopy codem = nn.BatchNorm2d(100)
input = torch.randn(20, 100, 35, 45)
output = m(input)
```

在这个例子中，`nn.BatchNorm2d(100)`定义了一个BatchNorm2d层，输入数据的通道数（num_features）是100。然后，它将对一个形状为[20, 100, 35, 45]的输入数据进行批量归一化处理。



## Coding Implementation

****

### `torchvision.transforms`

-   *用来做<u>图像数据增强</u>等**人工图像预处理**的库*
-   **不能做 SPL！！SPL 属于一种要自己搭建的模型内部的自主学习网络层* 
-   ==与 Transformer 模型无关！！！！！==

`torchvison.transforms` 包含了一些图像预处理操作，**这些操作可以使用`torchvison.transforms.Compose` 连在一起进行串行操作**（类似 `.nn.Sequential`用来串接网络层；`.nn.Sequential` 也能脚本化图像预处理操作，见 [Transforms Scripting](https://pytorch.org/vision/stable/transforms.html#transforms-scriptability)）。

**`torchvison.transforms`** 包含的主要操作：

```py
__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "Scale", "CenterCrop", "Pad",
           "Lambda", "RandomApply", "RandomChoice", "RandomOrder", "RandomCrop", "RandomHorizontalFlip", 
           "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation",           
           "ColorJitter", "RandomRotation", "RandomAffine", "Grayscale", "RandomGrayscale"]
```

-   `.Compose()`：用来排列集成所有的transforms操作。
-   `.ToTensor()`：把图片数据转换成张量并转化范围在[0,1]区间内。
-   `.Normalize(mean, std)`：归一化。
-   `.Resize(size)`：输入的PIL图像调整为指定的大小，参数可以为int或int元组。
-   `.CenterCrop(size)`：将给定的PIL Image进行中心切割，得到指定size的tuple。
-   `.RandomCrop(size, padding=0)`：随机中心点切割。
-   `.RandomHorizontalFlip(size, interpolation=2)`：将给定的PIL Image随机切割，再resize。
-   `.RandomHorizontalFlip()`：随机水平翻转给定的PIL Image。
-   `.RandomVerticalFlip()`：随机垂直翻转给定的PIL Image。
-   `.ToPILImage()`：将 Tensor 或 `numpy.ndarray` 转换为PIL Image。
-   `.FiveCrop(size)`：将给定的PIL图像裁剪成4个角落区域和中心区域。
-   `.Pad(padding, fill=0, padding_mode=‘constant’)`：对PIL边缘进行填充。
-   `.RandomAffine(degrees, translate=None, scale=None)`：保持中心不变的图片进行随机仿射变化。
-   `.RandomApply(transforms, p=0.5)`：随机选取变换。

自定义集成 transforms 的框架：

```python
My_Transforms = transforms.Compose([
    transforms.Resize((size, size)),  # simply resizing the image to manually fixed size
    # You may add some transforms here.
    transforms.ToTensor()  # simply transforming into Tensor (usually MUST be the last one of the transforms!!)
])
```

关于 `torchvision.transforms.functional` （transforms **函数**）：

`torchvision.transforms.functional` 模块是一组函数，它们对图像进行低级变换。**这些函数与`torchvision.transforms`模块中的类相对应，但以函数形式提供**。

例如，我们可以使用 `torchvision.transforms.Resize` 类来创建一个调整图像大小的变换对象，然后我们可以将图像传递给这个对象以调整其大小。另一方面，我们可以直接使用 `torchvision.transforms.functional.resize` 函数来调整图像的大小。

这个模块通常在你需要对图像**执行更灵活或更具体的变换**时使用。

****
