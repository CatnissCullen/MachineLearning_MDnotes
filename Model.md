# Model 

***万物皆可学习！（数据处理函数、距离度量函数、损失函数、模型本身 ......）***



## Quick Understanding of Deep Learning

-   Trainer: Us

-   Student: Machine (Computer)

-   **Plan before training (Need to be appropriate for the applicaion and easy to learn):** 

    a Function with Parameters to be Learnt & some Hyper-parameters (MANUALLY PROVIDED) => $Building\ the\ Model$

-   **Data for learning:** 

    a Training Set => $Train\ Time$

-   **Notes summarizing the learning:** 

    the Function with learned Parameters => $Learnt\ Model$

    ==To come up with the right Function is the key stage of Deep Learning!!!==

-   **Data for checking learning effect (Homework):**  

    a Validation Set => $Validation\ Time$

-   **Errors Recording:**

    Loss Function => *We tell our machine how bad the model is with this (Usually include <u>penalty \<Regularization Loss></u> in case of over-fitting T-T; Encourage simpler model so it needs to overcome the Regularization to attain more complex model, which will be more robust than complex ones without penalty)* 

-   **Consolidate the knowledge:** 

    the Training-Validation Loop => $Optimization$  

-   **==Data for the FINAL EXAM:==** 

    ==a Testing Set unseen through any stages before => $Test\ Time$==



## Quick Understanding of Neural Networks

**Basic Form of a $N+1$ Layers NN Model:** 

-   ==A **`Layer`**==   

    -   **`Affine Net`** ➡️

        >   ### Affine Transformation in Neural Networks
        >
        >   1.  **Linear Transformation**: This involves multiplying the input by a weight matrix. If you have an input vector $X$ and a weight matrix $W$, the linear transformation is $X⋅W$.
        >   2.  **Translation (Bias Addition)**: After the linear transformation, a bias vector is added to the result. If the bias vector is $b$, the full affine transformation is $X⋅W+b$.
        >
        >   Affine Layers are usually **FC** Layers.

    -   *(OPTIONAL)* **`Batch Normalization`** or **`Layer Normalization`** ➡️

    -   **`Non-Linear Activation Func`** (Like a firing rate of impulses carried away from cell body; most similar one to the actual brain is the **`ReLU`**, which is most commonly used) ➡️

    -   *(OPTIONAL)* **`Dropout`** ➡️

-   **Stack " a `Layer`" $N$ times until it's a complex enough non-linear function (but need to trade-off with difficulty in tackling overfitting)** ➡️
-   The **`Final Layer`**
    -   **`Affine Net`**➡️
    -   (OPTIONAL, <u>NEEDED IN CLASSIFICATION</u>) **`Softmax`**

<img src="images/image-20230819180311872.png" alt="image-20230819180311872" style="zoom: 25%;" />

![2e775eb44e3a0edeb4debd1f3a309cb](images/2e775eb44e3a0edeb4debd1f3a309cb.jpg)

![image-20230819181131837](images/image-20230819181131837.png)

<img src="images/image-20230819181020242.png" alt="image-20230819181020242" style="zoom:50%;" />

<img src="images/image-20230819181206616.png" alt="image-20230819181206616" style="zoom:50%;" />





## Activation Function

1.  **Sigmoid**:

    -   **Function**: 
        $$
        f(x)=\frac{1}{1+e^{-x}}
        $$
        
-   **Gradient**: 
        $$
        f'(x)=f(x)(1-f(x))
        $$
        
    -   **Problematic Region**: The gradient becomes close to zero for very large negative or positive values of $x$. In these regions, the sigmoid function saturates, meaning that it becomes very flat. This leads to vanishing gradients during backpropagation, which can slow down or halt training.

2.  **ReLU (Rectified Linear Unit)**:

    -   **Function**:
        $$
        f(x)=max(0,x)
        $$
        
-   **Gradient**: 
        $$
        f'(x)=\{^{1\quad if\ x>0}_{0\quad if\ x\le0}
        $$
        
    -   **Problematic Region**: The gradient is exactly zero for negative inputs. This can lead to "dead neurons" where once a neuron gets a negative input, it always outputs zero, and its weights never get updated. This is known as the dying ReLU problem.

3.  **Leaky ReLU**:

    -   **Function**: 
        $$
        f(x)=\{^{x\quad if\ x>0}_{\alpha x\quad if\ x\le0}
        $$
        
    -   **Gradient**: 
        $$
        f'(x)=\{^{1\quad if\ x>0}_{\alpha\quad if\ x\le0}
        $$
        
    -   **Problematic Region**: Leaky ReLU attempts to fix the dying ReLU problem by having a small positive gradient for negative inputs. This means that the gradient is never exactly zero, but if $\alpha$ is very small, the gradient can still be close to zero for negative inputs, potentially slowing down training.

    ***COMPARiSON***:

    -   **Sigmoid** has vanishing gradient problems for very large negative or positive inputs.
    -   **ReLU** has zero gradient for negative inputs, leading to the dying ReLU problem.
    -   **Leaky ReLU** attempts to mitigate the dying ReLU problem but can still have near-zero gradients for negative inputs if $α$ is very small.

    

4.  The **Hyperbolic Tangent (tanh)** is an activation function used in neural networks, including RNNs. It's defined as:
    $$
    tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}
    $$
    The function maps any real-valued number to the range −1,1−1,1. Here's what it looks like:

    Graph of tanh(x)Graph of tanh(x)

    The tanh function is zero-centered, meaning that negative inputs will be mapped strongly negative and zero inputs will be near zero in the output. This makes it easier for the model to learn from the backpropagated error and can result in faster training.

    Here are some properties of the tanh activation function:

    1.  **Non-linear**: This allows the model to learn from the error and make adjustments, which is essential for learning complex patterns.
    2.  **Output range**: The output values are bound within the range −1−1 and 11, providing normalized outputs.
    3.  **Zero-centered**: This helps mitigate issues related to the gradients and speeds up the training process.

    ***COMPARISON:***

    ### ReLU (Rectified Linear Unit)

    1.  **Computational Efficiency**: ReLU is computationally cheaper to calculate than tanh because it doesn't involve any exponential operations. This makes it faster to train large networks.
    2.  **Sparsity**: ReLU activation leads to sparsity. When the output is zero, it's essentially ignoring that neuron, leading to a sparse representation. Sparsity is beneficial because it makes the network easier to optimize.
    3.  **Non-vanishing Gradients**: ReLU doesn't suffer from the vanishing gradient problem for positive values, which makes it suitable for deep networks.

    ### tanh (Hyperbolic Tangent)

    1.  **Zero-Centered**: Unlike ReLU, tanh is zero-centered, making it easier for the model to learn in some cases.
    2.  **Output Range**: The output range of tanh is [−1,1][−1,1], which can be more desirable than [0,∞)[0,∞) for ReLU in certain applications like RNNs.
    3.  **Vanishing Gradients**: tanh can suffer from vanishing gradients for very large or very small input values, which can slow down learning.

    ### Context-Specific Usage

    -   **RNNs**: tanh is often used because the zero-centered nature of the function can be beneficial for maintaining the state over time steps.
    -   **CNNs and Fully-Connected Networks**: ReLU is often preferred due to its computational efficiency and because CNNs often deal with larger and deeper architectures where vanishing gradients are less of a concern.



## Hyper-Parameters

### Common Ones

-   **Batch Size**

    >   *It is usually based on memory constraints (if any), or set to some value, e.g. 32, 64 or 128. We **use powers of 2 in practice** because many vectorized operation implementations work faster when their inputs are sized in powers of 2.*

-   **(Initial) Learning Rate (set in Optimizer)** 

    >   ***Effect of step size**. Choosing the step size (also called the learning rate) will become one of the most important (and most headache-inducing) hyperparameter settings in training a neural network. In our blindfolded hill-descent analogy, <u>we feel the hill below our feet sloping in some direction, but the step length we should take is uncertain</u>. If we shuffle our feet carefully we can expect to make consistent but very small progress (this corresponds to having a small step size). Conversely, we can choose to make a large, confident step in an attempt to descend faster, but this may not pay off. At some point taking a bigger step gives a higher loss as we “overstep”.*

-   **Momentum (set in Optimizer)**

-   **Dropout (set in NN Module)**

-   **Weight-decay (set in Optimizer)**

### How to Tune

#### General

Initialize a set of HP -> Train on training set with this set of HP -> Get the best learned Params set (the Model) of this set of HP -> Predict with validation set -> Get an Acc -> ... (<u>**CROSS -VALIDATION**</u> LOOP UNTIL THE ACC IS SATISFYING ENOUGH)

#### Tips

>   ### Hyperparameter optimization
>
>   As we’ve seen, training Neural Networks can involve many hyperparameter settings. The most common hyperparameters in context of Neural Networks include:
>
>   -   the initial learning rate
>   -   learning rate decay schedule (such as the decay constant)
>   -   regularization strength (L2 penalty, dropout strength)
>
>   But as we saw, there are many more relatively less sensitive hyperparameters, for example in per-parameter adaptive learning methods, the setting of momentum and its schedule, etc. In this section we describe some additional tips and tricks for performing the hyperparameter search:
>
>   **Implementation**. Larger Neural Networks typically require a long time to train, so performing hyperparameter search can take many days/weeks. It is important to keep this in mind since it influences the design of your code base. One particular design is to have a **worker** that continuously samples random hyperparameters and performs the optimization. During the training, the worker will keep track of the validation performance after every epoch, and writes a model checkpoint (together with miscellaneous training statistics such as the loss over time) to a file, preferably on a shared file system. It is useful to include the validation performance directly in the filename, so that it is simple to inspect and sort the progress. Then there is a second program which we will call a **master**, which launches or kills workers across a computing cluster, and may additionally inspect the checkpoints written by workers and plot their training statistics, etc.
>
>   **Prefer one validation fold to cross-validation**. In most cases a single validation set of respectable size substantially simplifies the code base, without the need for cross-validation with multiple folds. You’ll hear people say they “cross-validated” a parameter, but many times it is assumed that they still only used a single validation set.
>
>   **Hyperparameter ranges**. Search for hyperparameters on log scale. For example, a typical sampling of the learning rate would look as follows: `learning_rate = 10 ** uniform(-6, 1)`. That is, we are generating a random number from a uniform distribution, but then raising it to the power of 10. The same strategy should be used for the regularization strength. Intuitively, this is because learning rate and regularization strength have multiplicative effects on the training dynamics. For example, a fixed change of adding 0.01 to a learning rate has huge effects on the dynamics if the learning rate is 0.001, but nearly no effect if the learning rate when it is 10. This is because the learning rate multiplies the computed gradient in the update. Therefore, it is much more natural to consider a range of learning rate multiplied or divided by some value, than a range of learning rate added or subtracted to by some value. Some parameters (e.g. dropout) are instead usually searched in the original scale (e.g. `dropout = uniform(0,1)`).
>
>   **Prefer random search to grid search**. As argued by Bergstra and Bengio in [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf), “randomly chosen trials are more efficient for hyper-parameter optimization than trials on a grid”. As it turns out, this is also usually easier to implement.
>
>   <img src="images/gridsearchbad.jpeg" alt="img" style="zoom:50%;" />
>
>   Core illustration from [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) by Bergstra and Bengio. It is very often the case that some of the hyperparameters matter much more than others (e.g. top hyperparam vs. left one in this figure). Performing random search rather than grid search allows you to much more precisely discover good values for the important ones.
>
>   **Careful with best values on border**. Sometimes it can happen that you’re searching for a hyperparameter (e.g. learning rate) in a bad range. For example, suppose we use `learning_rate = 10 ** uniform(-6, 1)`. Once we receive the results, it is important to double check that the final learning rate is not at the edge of this interval, or otherwise you may be missing more optimal hyperparameter setting beyond the interval.
>
>   **Stage your search from coarse to fine**. In practice, it can be helpful to first search in coarse ranges (e.g. 10 ** [-6, 1]), and then depending on where the best results are turning up, narrow the range. Also, it can be helpful to perform the initial coarse search while only training for 1 epoch or even less, because many hyperparameter settings can lead the model to not learn at all, or immediately explode with infinite cost. The second stage could then perform a narrower search with 5 epochs, and the last stage could perform a detailed search in the final range for many more epochs (for example).
>
>   **Bayesian Hyperparameter Optimization** is a whole area of research devoted to coming up with algorithms that try to more efficiently navigate the space of hyperparameters. The core idea is to appropriately balance the exploration - exploitation trade-off when querying the performance at different hyperparameters. Multiple libraries have been developed based on these models as well, among some of the better known ones are [Spearmint](https://github.com/JasperSnoek/spearmint), [SMAC](http://www.cs.ubc.ca/labs/beta/Projects/SMAC/), and [Hyperopt](http://jaberg.github.io/hyperopt/). However, in practical settings with ConvNets it is still relatively difficult to beat random search in a carefully-chosen intervals. See some additional from-the-trenches discussion [here](http://nlpers.blogspot.com/2014/10/hyperparameter-search-bayesian.html).
>
>   --- cs231n



## Evaluation (of Generalization) & Solutions

在统计学和机器学习中，**bias（偏差）**和**variance（方差）**是两个重要的概念，它们是用来<u>衡量模型预测**误差的主要来源**</u>。在设计和训练机器学习模型时，存在一个bias-variance权衡的问题。如果**模型过于简单并且具有有限的参数（模型的特征太弱，不足以区分不同样本，对样本变化不敏感），则可能会有高偏差和低方差。在另一方面，如果模型过于复杂并且具有大量的参数（模型的特征太强，过于区分不同样本，对样本变化太敏感），则可能会有低偏差和高方差。**在这两种情况下，模型在测试集上的表现都可能不佳。因此，我们的目标是**找到偏差和方差之间的最佳平衡点，以实现最佳的泛化能力**。

1.  **Bias（偏差）**：偏差是指模型的预测值与实际值的平均差异，或者说是模型的预期预测与真实预测之间的差距。简单来说，高偏差模型会忽视数据中的细节，这通常会导致欠拟合（underfitting），即模型在训练集和测试集上的表现都不好。

    <img src="images/image-20230724173953912.png" alt="image-20230724173953912" style="zoom:50%;" />

    *($\mu$的一阶矩是无偏估计)*  

2.  **Variance（方差）/ Deviation（标准差）**：方差是指模型预测的变化性或离散程度，即同样大小的不同训练集训练出的模型的预测结果的变化。如果模型对训练数据的小变动非常敏感，那么模型就有高方差，这通常会导致过拟合（overfitting），即模型在训练集上表现很好，但在测试集（即未见过的数据）上表现差。

    <img src="images/image-20230724174527318.png" alt="image-20230724174527318" style="zoom:50%;" />

    *（$\sigma$的一阶矩是有偏估计）*

    找同一模型公式的最佳参数相当于对着靶子开一枪，**偏差**相当于瞄准点偏离了实际靶心，**方差**相当于射击点偏离了瞄准点：

    <img src="images/image-20230724174659994.png" alt="image-20230724174659994" style="zoom: 33%;" />

    <img src="images/image-20230724175124356.png" alt="image-20230724175124356" style="zoom: 40%;" />

    （不同的 $f$ 对应用不同数据点训练出的同一模型公式的不同模型参数）

    不同公式的模型的训练结果分布图：

    <img src="images/image-20230724175523663.png" alt="image-20230724175523663" style="zoom:40%;" />

    越复杂的模型受数据集变化的影响越大，变动幅度越大，不同 “universe” 的训练结果间差距更大，即方差更大：

    <img src="images/image-20230724175711069.png" alt="image-20230724175711069" style="zoom:40%;" />

    但方差大时覆盖的范围也越大，取均值（一阶矩）时更容易覆盖“靶心”，因此越复杂的模型偏差越小：

    <img src="images/image-20230724180100173.png" alt="image-20230724180100173" style="zoom:40%;" />

    **结论：**复杂模型（参数多）最理想优化结果和真实情况间 Loss （即Bias）更小，但实际的 training 结果和理想情况差距很大，因为参数多意味着函数空间更大更难优化，需要更大的 Data Set。

<img src="images/image-20230724180242714.png" alt="image-20230724180242714" style="zoom: 33%;" />

<img src="images/image-20230724180355587.png" alt="image-20230724180355587" style="zoom: 33%;" />

<img src="images/image-20230724180533868.png" alt="image-20230724180533868" style="zoom: 33%;" />

<img src="images/image-20230724180635703.png" alt="image-20230724180635703" style="zoom:40%;" />

Should：（把所有训练集划分成训练集、验证集）① 用训练集训练 【取训练集对应的最佳参数】-> ② 用验证集再训练 【取①以后验证集对应的最佳参数】-> ③ 用所有数据集最后再训练【取②以后所有数据集对应的最佳参数】

<img src="images/image-20230724180852393.png" alt="image-20230724180852393" style="zoom:40%;" />

<img src="images/image-20230724181425216.png" alt="image-20230724181425216" style="zoom:40%;" />

**虽然直接优化目标是Loss小，而非偏差、方差是否最佳平衡（这也很难作为优化目标），**<u>但一般训练集Loss足够小且测试集Loss不严重高于训练集，就是比较平衡的结果</u>。

**解决模型复杂度（参数多导致的巨大函数空间）和与真实值的偏差之间的矛盾**：<u>用多层级的神经网络 =》**深度学习（见 Why Deep?????????）**</u>

<img src="images/image-20230724234749795.png" alt="image-20230724234749795" style="zoom: 33%;" />



## Deal With Over-fitting

***预测结果不好（准确度/ 损失值）但训练结果更好 —— 过拟合***

![image-20230812120700025](images/image-20230812120700025.png)

### Early Stopping

准备验证集（一般是所有已知数据**训练:验证=8:2**），一边训练以更新参数一边在每个epoch结束用验证集求误差（训练误差也求出来，用于比较观察是否过拟合）：

-   以验证集误差小于某阈值或更新多少次后不再出现更小值为 early stopping 标志；
-   以验证集误差在多少次后不再小于等于训练集误差为标志。

期间保留<u>使验证集误差最小</u>的参数直到最后。

### Dropout

***detail see https://cs231n.github.io/neural-networks-2/#reg***

类似决策树的剪枝，但 Dropout 一般是按比例随机关闭神经元（每个$W$中的某些`input_dim`的向量\<一整列设为0>），剪枝是经泛化性能的比较后减去决策分支

>   **Dropout** is an extremely effective, simple and recently introduced regularization technique by Srivastava et al. in [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) (pdf) that complements the other methods (L1, L2, maxnorm). While training, dropout is implemented by only keeping a neuron active with some probability $p$ (a hyperparameter), or setting it to zero otherwise.
>
>   --- *cs231n*

![image-20230825153900438](images/image-20230825153900438.png)

简单做法：单纯在训练时以 $p$ 的概率关闭神经元训练，但由于这样训练后每个神经元的输出为 
$$
out'=p\cdot out+(1-p)\cdot0
$$
所以用完整网络验证/测试时，为了发挥出用 $Dropout$ 训练的模型的性能（达到输出为 $p\cdot out+(1-p)\cdot0$ 的效果），**需要把原始输出层全部缩小到 $p$ 倍：**

```python
""" Vanilla Dropout: Not recommended implementation """

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  """ X contains the data """
  
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3
```

✨ 优解：直接在训练使用 $Dropout$ 的同时**将输出层扩大到 $1/p$ 倍**，验证/测试时无需任何改动，更方便：

```python
""" 
Inverted Dropout: Recommended implementation example.
We drop and scale at train time and don't do anything at test time.
"""

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3
  
  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)
  
def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```

### Shuffle

每个epoch开始前对batches洗牌（在 `DataLoader` 处设置，训练集和验证集开启，测试集关闭）

### Maximize the Margin + Soft Margin

在分类问题中，最大化有训练集学得的划分超平面两侧的间隔（参考《机器学习》支持向量机），并且允许部分训练数据出错，即落入间隔内而不在硬间隔外部的严格分类区域（软化间隔）。

### Max norm constraints 

>   Another form of regularization is to enforce an absolute upper bound on the magnitude of the weight vector for every neuron and use projected gradient descent to enforce the constraint. In practice, this corresponds to performing the parameter update as normal, and then **enforcing the constraint by clamping the weight vector $\vec W$ of every neuron to satisfy $||\vec W||_2<c$. Typical values of $c$ are on orders of 3 or 4.** Some people report improvements when using this form of regularization. One of its appealing properties is that **network cannot “explode” even when the learning rates are set too high because the updates are always bounded.**
>
>   --- *cs231n*

### Support Vector Regression (SVR)

在回归问题中，容忍落在间隔带内的出错数据，不计算其误差到 Loss ，即给 Loss Function 添加一个不敏感损失函数项，类似正则化（参考《机器学习》支持向量机）。

### L2 Regularization (Weight-decay)

See [Loss Function](D:\CAMPUS\AI\MachineLearning\LossFunction.md)

***正则项本质上是支持向量机的最大化间隔法产生的，其表示的其实是划分超平面两侧安全间隔的总大小（待最大化），参考《机器学习》p122（侧栏笔记有推导）,123,133*** 

### L1 Regularization

比 $L2$ 更鼓励 $W$ 为稀疏矩阵，正则项为 $W$ 的 $L1$ 距离

### (In Classification) Weaken Features of a Each Class

估计各类别的分布时（建模时），**使它们拥有相同的某参数**，该参数<u>由各类别分别用对应训练集估计出的参数再对样本数**加权平均**得到</u>

### Simplify Network Structure to Cut Down Sensitivity of  Features' Alternation

See **<u>CNN</u>**

### Batch Normalization

See  [DataProcessing](D:\CAMPUS\AI\MachineLearning\ML_MDnotes\DataProcessing.md)---**<u>Batch Normalization</u>**

### Cross Validation, K-Fold

用于调节超参数的验证方法



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

***（详见《机器学习》第8章）***

**"Ensemble"（集成）**是一种机器学习策略，它结合了多个模型的预测，以产生一个最终的预测结果。这种方法的主要目标是通过组合多个模型来减小单一模型的预测误差。

主要的集成方法有：

1.  Bagging（Bootstrap Aggregating）：通过从训练数据中<u>**有放回地随机抽样（自助采样法）**来生成不同的训练集，然后在每个训练集上训练一个模型</u>。最后，所有模型的预测结果**通过投票**（多数胜出；分类问题）或**平均**（回归问题）来产生最终预测。**随机森林（Random Forest）**就是一个典型的Bagging的例子。

2.  **Boosting**：一种迭代的策略，其中新模型的训练依赖于之前模型的性能。每个新模型都试图修正之前模型的错误。最著名的Boosting算法有AdaBoost和梯度提升。

    >   *先从初始训练集训练出一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的训练样本在后续受到更多关注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至基学习器数目达到事先指定的值 $T$，最终将这 $T$ 个基学习器进行加权结合。*
    >
    >   *—— 《机器学习》P173*
    >
    >   ***调整样本分布包括<u>重赋权法（适用于样本可带权学习算法、学习过程未因遇到不满足基本条件的基学习器而提前停止的情况）</u>和<u>重采样法（适用于不可带权学习算法、学习过程因遇到不满足基本条件的基学习器而提前停止的情况）</u>***

3.  Stacking：使用多个模型来训练数据，然后再用一个新的模型（叫做元模型或者二级模型）来预测这些模型的预测结果。

在许多机器学习竞赛中，集成学习被证明是一种非常有效的方法，因为它可以**减少过拟合，增加模型的泛化能力，从而提高预测性能**。

>   In practice, one reliable approach to improving the performance of Neural Networks by a few percent is to train multiple independent models, and at test time average their predictions. As the number of models in the ensemble increases, the performance typically monotonically improves (though with diminishing returns). Moreover, the improvements are more dramatic with higher model variety in the ensemble. There are a few approaches to forming an ensemble:
>
>   -   **Same model, different initializations**. Use cross-validation to determine the best hyperparameters, then train multiple models with the best set of hyperparameters but with different random initialization. The danger with this approach is that the variety is only due to initialization.
>   -   **Top models discovered during cross-validation**. Use cross-validation to determine the best hyperparameters, then pick the top few (e.g. 10) models to form the ensemble. This improves the variety of the ensemble but has the danger of including suboptimal models. In practice, this can be easier to perform since it doesn’t require additional retraining of models after cross-validation
>   -   **Different checkpoints of a single model**. If training is very expensive, some people have had limited success in taking different checkpoints of a single network over time (for example after every epoch) and using those to form an ensemble. Clearly, this suffers from some lack of variety, but can still work reasonably well in practice. The advantage of this approach is that is very cheap.
>   -   **Running average of parameters during training**. Related to the last point, a cheap way of almost always getting an extra percent or two of performance is to maintain a second copy of the network’s weights in memory that maintains an exponentially decaying sum of previous weights during training. This way you’re averaging the state of the network over last several iterations. You will find that this “smoothed” version of the weights over last few steps almost always achieves better validation error. The rough intuition to have in mind is that the objective is bowl-shaped and your network is jumping around the mode, so the average has a higher chance of being somewhere nearer the mode.
>
>   One disadvantage of model ensembles is that they take longer to evaluate on test example. An interested reader may find the recent work from Geoff Hinton on [“Dark Knowledge”](https://www.youtube.com/watch?v=EK61htlw8hY) inspiring, where the idea is to “distill” a good ensemble back to a single model by incorporating the ensemble log likelihoods into a modified objective.
>
>   --- *cs231n*



## Why Deep?????????

-   在比浅层模型的各层参数个数少的情况下，通过增加深度来增加模型复杂度以减小偏差，且最终整体参数个数也少于浅层但各层参数更多的模型【复杂化过程对参数的利用率更高】，这样训练需要的数据也更少（在真实情况复杂又有潜在规律\<比如语音、图像的模型>时效果更突出！！！）
-   利用多层激活函数逐层引入非线性逼近复杂的函数



## Classic Layers/ Model Structures

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

https://cs231n.github.io/convolutional-networks/#conv

<img src="images/y(1).png" alt="y(1)" style="zoom: 80%;" /> 

#### Basic Form

==***INPUT -> [[CONV -> RELU]\*N -> POOL?]\*M -> [FC -> RELU]\*K -> FC**==  

>   We use three main types of layers to build ConvNet architectures: **Convolutional Layer**, **Pooling Layer**, and **Fully-Connected Layer** (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet **architecture**.
>
>   *A ConvNet is made up of Layers. Every Layer has a simple API: It transforms an input 3D volume to an output 3D volume with some differentiable function that **may or may not have parameters (e.g. CONV/FC do, RELU/POOL don’t)** and also **may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn’t)**.*  
>
>   --- *cs231n*

<img src="images/image-20230901104723724.png" alt="image-20230901104723724" style="zoom:30%;" />

<img src="images/image-20230901144201391.png" alt="image-20230901144201391" style="zoom: 50%;" />

#### Usage Overview

-   ***从网络内部结构改变模型性能***

-   专用于图像数据（3维矩阵），<u>网络内部保持3维的矩阵格式</u>

-   需要前置 **Spatial Transformer Layers** 

-   据具体问题考虑是否使用 **Pooling**（不是所有数据都能用！！一般图像可以）

-   可以添加 **Residual Block：** ***【RNN也可以用】通过将输入 x 直接加到输出 F(x) 添加梯度捷径***

    ![image-20230909113143212](images/image-20230909113143212.png)

    **作用：① 当权重消失时输出仍有 x ，因此若某层网络是冗余的，即当损失值很大时（尤其加入正则项后），在学习过程中权重会变小但仍保留先前输出，不会给整个模型带来多余负面影响（相当于关闭不需要的网络层原封不动输出其输入）；② 反向传播时提供了更通畅的偏导支路（类似高速公路的作用），加速网络 converge v 速度**

-   其他CNN结构：

    ![image-20230909115203190](images/image-20230909115203190.png)

-   

事实上也可以用CNN指向Fully-Connected Layer前面的部分（只包括Convolutional Layer和Pooling Layer）

**IMPORTANT TIPS:**

-   ***Prefer a stack of small filter CONV to one large receptive field CONV layer***

    >   Suppose that you stack three 3x3 CONV layers on top of each other (with non-linearities in between, of course). In this arrangement, each neuron on the first CONV layer has a 3x3 view of the input volume. A neuron on the second CONV layer has a 3x3 view of the first CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. Second, if we suppose that all the volumes have $C$ channels, then it can be seen that the single 7x7 CONV layer would contain $C×(7×7×C)=49C^2$ parameters, while the three 3x3 CONV layers would only contain $3×(C×(3×3×C))=27C^2$ parameters. Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. <u>As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.</u>
    >
    >   

-   ***Use Pretrained CNN from ImageNet for work!!!!!!!*** 

    >   **Recent departures.** It should be noted that the conventional paradigm of a linear list of layers has recently been challenged, in Google’s Inception architectures and also in current (state of the art) Residual Networks from Microsoft Research Asia. Both of these (see details below in case studies section) feature more intricate and different connectivity structures.
    >
    >   **In practice: use whatever works best on ImageNet**. If you’re feeling a bit of a fatigue in thinking about the architectural decisions, you’ll be pleased to know that in 90% or more of applications you should not have to worry about these. I like to summarize this point as “*don’t be a hero*”: Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your data. You should rarely ever have to train a ConvNet from scratch or design one from scratch. I also made this point at the [Deep Learning school](https://www.youtube.com/watch?v=u6aEYuemt0M).
    >
    >   --- *cs231n*

-   ***Input Size:***

    >   The **input layer** (that contains the image) should be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.
    >
    >   --- *cs231n*

-   ***Conv Layer Size:***

    >   The **conv layers** should be using small filters (e.g. 3x3 or at most 5x5), using a stride of S=1, and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when F=3, then using P=1 will retain the original size of the input. When F=5, P=2. For a general F, it can be seen that P=(F−1)/2 preserves the input size. If you must use bigger filter sizes (such as 7x7 or so), it is only common to see this on the very first conv layer that is looking at the input image.
    >
    >   --- *cs231n*

-   ***Pool Layer Size:***

    >   The **pool layers** are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. F=2), and with a stride of 2 (i.e. S=2). Note that this discards exactly 75% of the activations in an input volume (due to downsampling by 2 in both width and height). Another slightly less common setting is to use 3x3 receptive fields with a stride of 2, but this makes “fitting” more complicated (e.g., a 32x32x3 layer would require zero padding to be used with a max-pooling layer with 3x3 receptive field and stride 2). It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and aggressive. This usually leads to worse performance.
    >
    >   --- *cs231n*

-   ***Reducing sizing headaches:*** 

    >   The scheme presented above is pleasing because all the CONV layers preserve the spatial size of their input, while the POOL layers alone are in charge of down-sampling the volumes spatially. In an alternative scheme where we use strides greater than 1 or don’t zero-pad the input in CONV layers, we would have to very carefully keep track of the input volumes throughout the CNN architecture and make sure that all strides and filters “work out”, and that the ConvNet architecture is nicely and symmetrically wired.
    >
    >   --- *cs231n*

-   ***Why use stride of 1 in CONV?*** 

    >   Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.
    >
    >   --- *cs231n*

-   ***Why use padding?***

    >   In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. <u>If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly.</u>
    >
    >   --- *cs231n*

    ***虽然padding补的是0，但是这样可以使输入图像的边缘也有机会充分接触 Filter 的各处权重***

-    ***Compromising based on <u>memory constraints</u>:***

    >   In some cases (especially early in the ConvNet architectures), the amount of memory can build up very quickly with the rules of thumb presented above. For example, filtering a 224x224x3 image with three 3x3 CONV layers with 64 filters each and padding 1 would create three activation volumes of size [224x224x64]. This amounts to a total of about 10 million activations, or 72MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the first CONV layer of the network. For example, one compromise might be to use a first CONV layer with filter sizes of 7x7 and stride of 2 (as seen in a ZF net). As another example, an AlexNet uses filter sizes of 11x11 and stride of 4.
    >
    >   --- *cs231n*

-   ***Computational Considerations:***

    >   The largest bottleneck to be aware of when constructing ConvNet architectures is the memory bottleneck. Many modern GPUs have a limit of 3/4/6GB memory, with the best GPUs having about 12GB of memory. There are three major sources of memory to keep track of:
    >
    >   -   From the intermediate volume sizes: These are the raw number of **activations** at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. first Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below.
    >   -   From the parameter sizes: These are the numbers that hold the network **parameters**, their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so.
    >   -   Every ConvNet implementation has to maintain **miscellaneous** memory, such as the image data batches, perhaps their augmented versions, etc.
    >
    >   Once you have a rough estimate of the total number of values (for activations, gradients, and misc), the number should be converted to size in GB. Take the number of values, multiply by 4 to get the raw number of bytes (since every floating point is 4 bytes, or maybe by 8 for double precision), and then divide by 1024 multiple times to get the amount of memory in KB, MB, and finally GB. If your network doesn’t fit, a common heuristic to “make it fit” is to decrease the batch size, since most of the memory is usually consumed by the activations.
    >
    >   --- *cs231n*

#### Advantage

**简化网络使得非全连接，以减少网络层权重（参数）数，进而削弱网络的特征敏感度，防止过拟合**

#### A Constraint Rule to Hyper-params

设输入2D维数为$N^2$，RF的2D维数为$F^2$（$Hyper$），Padding宽度为$P$（$Hyper$），卷积步长为$S$（$Hyper$），输出2D维数为$N'^2$，各维数均须为正整数，则恒有：
$$
N'=\frac{N-F+2P}{S}+1
$$
![e0caa8c2216966d03a45982ffc50b10](images/e0caa8c2216966d03a45982ffc50b10-1693557035121-4.jpg)

#### Special Normalizations for CNN

![image-20230906213347227](images/image-20230906213347227.png)

#### Coding

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

#### Application

<img src="images/image-20230909115712333.png" alt="image-20230909115712333" style="zoom: 33%;" /><img src="images/image-20230909115825005.png" alt="image-20230909115825005" style="zoom: 33%;" />

<img src="images/image-20230909120120367.png" alt="image-20230909120120367" style="zoom: 33%;" /><img src="images/image-20230909115932081.png" alt="image-20230909115932081" style="zoom: 33%;" />

**循环神经网络（Recurrent Neural Network，RNN）**是一种专门处理**序列数据**（例如，时间序列数据\<音频>或文本）的神经网络（可以对语音分类，不用一般的分类模型）。RNN与普通的全连接神经网络和卷积神经网络不同，它能够**处理序列长度可变的数据**，<u>在处理每个元素时，它都会记住前面元素的信息</u>。

**训练时所有输入都给出，并行输入：**

<img src="images/image-20230909141309461.png" alt="image-20230909141309461" style="zoom:50%;" />

**测试时只提供第一步输入，在上一步输出的概率分布中抽样作为其状态的最终输出，并作为下一步的输入（改为one-hot），串行输入：**

<img src="images/image-20230909141728529.png" alt="image-20230909141728529" style="zoom:50%;" />

***此外也可以把非序列数据拆解成序列信息学习，进而优化学习效果：***

<img src="images/image-20230909120400575.png" alt="image-20230909120400575" style="zoom:50%;" />

<img src="images/image-20230909120458610.png" alt="image-20230909120458610" style="zoom:50%;" />

#### Layers & Structure

RNN的基本思想是在神经网络的**隐藏层之间建立循环连接**。**每一步都会有两个输入：当前步的输入数据和上一步的隐藏状态（于是每次model会有两个输出值，训练过程注意左值设为`output, _`）。**然后，这两个输入会被送入网络（通常是一个全连接层或者一些更复杂的结构，如LSTM或GRU单元），然后产生一个输出和新的隐藏状态。这个新的隐藏状态将被用于下一步的计算。

这个过程可以写作如下形式的**状态转移方程**：
$$
h_t = f(h_{t-1}, x_t) = h_{t-1}\cdot W_h + x_t\cdot W_x+bias
$$
其中，**$h_t$ 是在时间t的隐藏状态（H元向量）**，$x_t$ 是在时间t的输入（D元向量），$f$ 是一个线性函数，它定义了<u>如何用前一步的隐藏状态和当前的输入计算得到当前的隐藏状态</u>，其中状态权重 $W_h$ （H*H矩阵）、输入权重 $W_x$ （D*H矩阵）和偏差 $bias$ （H元向量）在各时刻（timestep）的**状态转移方程**中保持不变。

<img src="images/image-20230909120618449.png" alt="image-20230909120618449" style="zoom:50%;" />

通常还需要一层**非线性激活函数** —— $\tanh$：
$$
h_t = \tanh\ (h_{t-1}\cdot W_h + x_t\cdot W_x+bias)
$$
此时仅得到了新时刻的状态（通常先求出前向传播求出所有状态存在数组，再另外计算所有输出），**各状态下（时刻）的输出**还需另外根据该状态用得分权重 $W_y$ 计算，输出即不同可能取值的可能性得分（此后还需用 $softmax$ 转化为概率分布值）：
$$
y_t=W_y\cdot h_t
$$
***各种结构的RNN都满足以上公式，区别仅在于每一步的输入 $x$ 是否有、是否来自上次输出：***

<img src="images/image-20230909140656609.png" alt="image-20230909140656609" style="zoom:50%;" />

<img src="images/image-20230909140633758.png" alt="image-20230909140633758" style="zoom:50%;" />

<img src="images/image-20230909140546263.png" alt="image-20230909140546263" style="zoom:50%;" />

<img src="images/image-20230909141012278.png" alt="image-20230909141012278" style="zoom:50%;" />

**$f_W$是循环使用的！！！！所以称为循环神经网络**

#### Backprop

<img src="images/image-20230909142544735.png" alt="image-20230909142544735" style="zoom:50%;" />

<img src="images/image-20230909142650615.png" alt="image-20230909142650615" style="zoom:50%;" />

#### Porblems

<img src="images/image-20230909164318037.png" alt="image-20230909164318037" style="zoom:50%;" />

处理长序列时，它们往往会遇到所谓的梯度消失和梯度爆炸问题。这些问题已经有一些解决方案，例如**长短期记忆（Long Short-Term Memory，LSTM）网络**和**门控循环单元（Gated Recurrent Unit，GRU）**。

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

<img src="images/image-20230805110434586.png" alt="image-20230805110434586" style="zoom:50%;" />

<img src="images/image-20230909144016690.png" alt="image-20230909144016690" style="zoom:50%;" />

<img src="images/image-20230909144440206.png" alt="image-20230909144440206" style="zoom:50%;" />

<img src="images/image-20230909144520905.png" alt="image-20230909144520905" style="zoom:50%;" />

<img src="images/image-20230909144537625.png" alt="image-20230909144537625" style="zoom:50%;" />

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

****

### Generative Model

***（如何假设分布：多元特征样本则对训练集每类样本都分别设为一个高斯分布<可共用方差并共同计算似然函数>，无论是否是二元分类；单元特征样本且二元分布则直接对所有训练集设为一整个伯努利分布）***

#### GAN（生成式对抗模型）

https://github.com/hindupuravinash/the-gan-zoo

https://youtu.be/4OWp0wDu6Xw

https://youtu.be/jNY1WBb8l4U

https://youtu.be/MP0BnVH2yOo

https://youtu.be/wulqhgnDr7E

https://gwern.net/face

$D$ and $G$ play a minimax game in which $D$ tries to
maximize the probability it correctly classifies reals and fakes
($logD(x)$), and $G$ tries to minimize the probability that
$D$ will predict its outputs are fake ($log(1-D(G(z)))$).
From the paper, the GAN loss function is

$$$

\begin{align}\underset{G}{\text{min}} \underset{D}{\text{max}}V(D,G) = \mathbb{E}**_**{x\sim p**_**{data}(x)}\big*[logD(x)\big]* + \mathbb{E}**_**{z\sim p**_**{z}(z)}\big*[log(1-D(G(z)))\big]*\end{align}

$$$

In theory, the solution to this minimax game is where
$p_g = p_{data}$, and the discriminator guesses randomly if the
inputs are real or fake. However, the convergence theory of GANs is
still being actively researched and in reality models do not always
train to this point.

<img src="images/image-20230914111616723.png" alt="image-20230914111616723" style="zoom: 25%;" />

<img src="images/image-20230914111758626.png" alt="image-20230914111758626" style="zoom:50%;" />



****

### Linear Classifier （即把 Linear Regression 的模型用来分类，把分类当线性回归做）

$$
\vec S_i = W\cdot \vec x_i\ +\ \vec b
$$

$\vec S_i$ 是第 i 个样本在各个类别中获得的分数组成的向量**（Unnormalized probabilities of Classes）**，$W$ 待学习的矩阵，$\vec b$ 是待学习的偏差向量，与具体训练集中不同类别的样本的数量差异有关（学习时将平衡这样的差异）。

优化目标：使得真实label对应的类别获得的分数最高

<u>**SVM** 使用的模型就是线性模型（寻找划分超平面）</u>

****

### Logistic Regression (Discriminative Model; Softmax Classifier)

***（不需要自己假设分布！！！）***

<img src="images/image-20230812173316594.png" alt="image-20230812173316594" style="zoom:50%;" />

<img src="images/image-20230812172514738.png" alt="image-20230812172514738" style="zoom: 50%;" />

和 Linear Regression 的区别仅在<u>**最终**输出时**多加一层 Softmax**</u>（这是 Softmax 作为生成概率分布的用途时的用法，<u>若是别的用途则在隐藏层也会加</u>！！！！sigmoid是特殊的Softmax，二分类时直接用sigmoid；**Logit \<对数几率，log odds>** 狭义即指 sigmoid 的反函数，广义指<u>未经 Softmax 的输出</u>），用 Linear Classification 输出的分数生成 **Normalized probabilities of Classes**（属于 (0, 1)，取不到边界）

添加 Softmax 后因为有了 (0, 1) 的上下界，高分数（向1挤压）会相较低分数（向0挤压）变得更高

优化目标：使得真实label对应的类别获得的分数最高 => **即接近一（因为已经转化为概率分布了）**

<img src="images/image-20230812172829902.png" alt="image-20230812172829902" style="zoom:50%;" />

其中“1”对应交叉熵损失中的实际分布中概率，“log(...)”对应交叉熵损失中的估计分布中概率

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

#### SVM vs SOFTMAX

![img](images/svmvssoftmax.png)

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





## 2 Modes of Model (nn.Module) & ` with torch.no_grad()`:

****

|                                                              |     训练时`.train()`      |            验证/ 测试时`.eval()`             |
| :----------------------------------------------------------: | :-----------------------: | :------------------------------------------: |
| 添加**`with torch.no_grad():`**禁止在`.forward()`时存储反向传播要用的值（`.forward()`默认内置这一操作） |             ❌             |       ✔️（节省存储且加速`.forward()`）        |
|                  自动调整学习率（如果可以）                  |             ✔️             |                      ❌                       |
|         据概率（给定参数）随机关闭神经元（dropout）          |             ✔️             |                      ❌                       |
|              `.BatchNorm2d`（如果模型中设置了）              | 计算每个batch的均值和方差 | 使用在训练过程中计算得到的移动平均均值和方差 |

