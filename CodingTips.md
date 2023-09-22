# Coding Tips

****



## Params Initialization

>   We have seen how to construct a Neural Network architecture, and how to preprocess the data. Before we can begin to train the network we have to initialize its parameters.
>
>   **Pitfall: all zero initialization**. Lets start with what we should not do. Note that we do not know what the final value of every weight should be in the trained network, but with proper data normalization it is reasonable to assume that approximately half of the weights will be positive and half of them will be negative. A reasonable-sounding idea then might be to set all the initial weights to zero, which we expect to be the “best guess” in expectation. This turns out to be a mistake, because if every neuron in the network computes the same output, then they will also all compute the same gradients during backpropagation and undergo the exact same parameter updates. In other words, there is no source of asymmetry between neurons if their weights are initialized to be the same.
>
>   **Small random numbers**. Therefore, we still want the weights to be very close to zero, but as we have argued above, not identically zero. As a solution, it is common to initialize the weights of the neurons to small numbers and refer to doing so as *symmetry breaking*. The idea is that the neurons are all random and unique in the beginning, so they will compute distinct updates and integrate themselves as diverse parts of the full network. The implementation for one weight matrix might look like `W = 0.01* np.random.randn(D,H)`, where `randn` samples from a zero mean, unit standard deviation gaussian. With this formulation, every neuron’s weight vector is initialized as a random vector sampled from a multi-dimensional gaussian, so the neurons point in random direction in the input space. It is also possible to use small numbers drawn from a uniform distribution, but this seems to have relatively little impact on the final performance in practice.
>
>   *Warning*: It’s not necessarily the case that smaller numbers will work strictly better. For example, a Neural Network layer that has very small weights will during backpropagation compute very small gradients on its data (since this gradient is proportional to the value of the weights). This could greatly diminish the “gradient signal” flowing backward through a network, and could become a concern for deep networks.
>
>   **Calibrating the variances with 1/sqrt(n)**. One problem with the above suggestion is that the distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that we can normalize the variance of each neuron’s output to 1 by scaling its weight vector by the square root of its *fan-in* (i.e. its number of inputs). ==That is, **the recommended heuristic is to initialize each neuron’s weight vector as: `w = np.random.randn(n) / sqrt(n)`, where `n` is the number of its inputs.**== This ensures that all neurons in the network initially have approximately the same output distribution and empirically improves the rate of convergence.
>
>   The sketch of the derivation is as follows: Consider the inner product 
>
>   <img src="images/image-20230825152018535.png" alt="image-20230825152018535" style="zoom:40%;" />
>
>    between the weights $w$ and input $x$, which gives the raw activation of a neuron before the non-linearity. We can examine the variance of $s$:
>
>   <img src="images/image-20230825152120252.png" alt="image-20230825152120252" style="zoom: 25%;" />
>
>   where in the first 2 steps we have used [properties of variance](http://en.wikipedia.org/wiki/Variance). In third step we assumed zero mean inputs and weights, so 
>
>   <img src="images/image-20230825152145953.png" alt="image-20230825152145953" style="zoom:40%;" />
>
>   Note that this is not generally the case: For example ReLU units will have a positive mean. In the last step we assumed that all $w_i$, $x_i$ are identically distributed. From this derivation we can see that if we want $s$ to have the same variance as all of its inputs $x$, then during initialization we should make sure that the variance of every weight $w$ is $1/n$. And since $Var(aX)=a^2Var(X)$ for a random variable $X$ and a scalar $a$, this implies that we should draw from unit gaussian and then scale it by a=$\sqrt{1/n}$, to make its variance $1/n$. This gives the initialization `w = np.random.randn(n) / sqrt(n)`.
>
>   A similar analysis is carried out in [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) by $Glorot\ et\ al$. In this paper, the authors end up recommending an initialization of the form $Var(w)=2/(n_{in}+n_{out})$ where $n_{in}$, $n_{out}$ are the number of units in the previous layer and the next layer. This is based on a compromise and an equivalent analysis of the backpropagated gradients. A more recent paper on this topic, [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv-web3.library.cornell.edu/abs/1502.01852) by $He\ et\ al$., derives an initialization specifically for ReLU neurons, reaching the conclusion that the variance of neurons in the network should be 2.0/n. This gives the initialization `w = np.random.randn(n) * sqrt(2.0/n)`, and is the current recommendation for use in practice in the specific case of neural networks with ReLU neurons.
>
>   **Sparse initialization**. Another way to address the uncalibrated variances problem is to set all weight matrices to zero, but to break symmetry every neuron is randomly connected (with weights sampled from a small gaussian as above) to a fixed number of neurons below it. A typical number of neurons to connect to may be as small as 10.
>
>   **Initializing the biases**. It is possible and common to initialize the biases to be zero, since the asymmetry breaking is provided by the small random numbers in the weights. For ReLU non-linearities, some people like to use small constant value such as 0.01 for all biases because this ensures that all ReLU units fire in the beginning and therefore obtain and propagate some gradient. However, it is not clear if this provides a consistent improvement (in fact some results seem to indicate that this performs worse) and it is more common to simply use 0 bias initialization.
>
>   **In practice**, the current recommendation is to use ReLU units and use the `w = np.random.randn(n) * sqrt(2.0/n)`, as discussed in [He et al.](http://arxiv-web3.library.cornell.edu/abs/1502.01852).
>
>   **Batch Normalization**. A recently developed technique by $Ioffe$ and $Szegedy$ called [Batch Normalization](http://arxiv.org/abs/1502.03167) alleviates a lot of headaches with properly initializing neural networks by explicitly forcing the activations throughout a network to take on a unit gaussian distribution at the beginning of the training. The core observation is that this is possible because normalization is a simple differentiable operation. In the implementation, applying this technique usually amounts to insert the `BatchNorm `layer immediately after fully connected layers (or convolutional layers, as we’ll soon see), and before non-linearities. We do not expand on this technique here because it is well described in the linked paper, but note that it has become a very common practice to use Batch Normalization in neural networks. In practice networks that use Batch Normalization are significantly more robust to bad initialization. Additionally, batch normalization can be interpreted as doing preprocessing at every layer of the network, but integrated into the network itself in a differentiable manner. Neat!
>
>   --- *cs231n*



## Structure

$exp:$ 

-   `Solver` class

    -   variable structure:

        -   Optional Arguments:

            -   ```apl
                batch_size
                num_epochs
                optimizer  
                optim_config  # hyper-params for the specific optimizer
                lr_decay  # A scalar for learning rate decay; after each epoch the learning rate is multiplied by this value.
                print_every  # Integer; training losses will be printed every 'print_every' iterations.
                num_train_sample  # Number of training samples used to check training accuracy; set to None to use entire training set.
                num_val_sample  # Number of validation samples to use to check val accuracy; default is None, which uses the entire validation set.
                checkpoint_name  # If not None, then save model checkpoints here every epoch.
                ```

        -   `data`
            -   `X_train` `X_val`
            -   `y_train` `y_val`(ground truth labels)
        -   `model` (class object)
            -   `__init__`
            -   `forwarding`
                1.  **`net_forward` (a specific net structure)**
                2.  **`activation` (a specific activation function)**
                3.  `net_forward`
                4.  `activation`
                5.  ...
                6.  `net_n_forward` => compute `scores`
            -   `backprop`
                1.  **`loss_func` (a specific loss function)** => <u>if no `y` input, just do validation (compute `loss` then return);</u> else compute `loss` & `d_outn`
                2.  **`net_backward` (according to net structure)** => compute `d_Wn` & `d_bn` & `d_out(n-1)_actv`
                3.  `actv_backward` => compute `d_out(n-1)`
                4.  ...

    -   functions structure:

        -   `train`

            -   **`for e in range(num_epochs):`  (Epoch \* 1)** 

                1.  **`for b in range(nums_batches):`  (Batch * 1)**

                    1.  `step`
                        1.  random fetch batch with `batch_mask`
                        2.  `model.loss_func` => get `loss` & all `grads['d_?']`
                        3.  **params update (gradients descend)** 
                    2.  print `loss` or `acc` or don't

                2.  `save_checkpoint`

                    $exp:$

                    ```python
                    	def _save_checkpoint(self):
                    		if self.checkpoint_name is None:
                    			return
                    		checkpoint = {
                    			"model": self.model,
                    			"update_rule": self.update_rule,
                    			"lr_decay": self.lr_decay,
                    			"optim_config": self.optim_config,
                    			"batch_size": self.batch_size,
                    			"num_train_samples": self.num_train_samples,
                    			"num_val_samples": self.num_val_samples,
                    			"epoch": self.epoch,
                    			"loss_history": self.loss_history,
                    			"train_acc_history": self.train_acc_history,
                    			"val_acc_history": self.val_acc_history,
                    		}
                    		filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
                    		if self.verbose:
                    			print('Saving checkpoint to "%s"' % filename)
                    		with open(filename, "wb") as f:
                    			pickle.dump(checkpoint, f)
                    ```

                3.  check `train_acc` on all `X_train` 

                4.  check `val_acc` on all `X_val` (`validation`)

                5.  **keep track of the best model according to `val_acc` with `best_params`**

                6.  print training & validation `acc` or not

                7.  **update `learning_rate` with `lr_decay` or not**

-   `Predictor` class







## For Convenience

-   把偏差 $b$ 看作属性值为常数1的一项的权重，合并到权重矩阵的最右一列，并且给输入向量（属性值向量）多增加恒为一的一元

    <img src="images/image-20230815102933855.png" alt="image-20230815102933855" style="zoom: 33%;" />
    
    ```python
    # append the bias dimension of ones (i.e. bias trick) so that our model only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    ```
    
    

## For Precision (Numerical  Problem)

### Normalization Trick

Dividing large numbers can be numerically unstable, so it is important to use a normalization trick. 

>   Notice that if we multiply the top and bottom of the fraction by a constant C and push it into the sum, we get the following (mathematically equivalent) expression:
>
>   <img src="images/image-20230818161831663.png" alt="image-20230818161831663" style="zoom: 33%;" />
>
>   We are free to choose the value of C. This will not change any of the results, but we can use this value to improve the numerical stability of the computation. A common choice for C is to set 
>
>   <img src="images/image-20230818161921151.png" alt="image-20230818161921151" style="zoom:33%;" />
>
>   This simply states that we should shift the values inside the vector f so that the highest value is zero. In code:
>
>   ```python
>   f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
>   p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup
>   
>   # instead: first shift the values of f so that the highest number is 0:
>   f -= np.max(f) # f becomes [-666, -333, 0]
>   p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
>   ```

### Log-Exponential Trick

```python
# Replace
x = pow(y, z)
# With
x = torch.exp(z * math.log(y))
```



## To Debug

-   用较小的值（≈ 0）或别的特殊值初始化模型参数使得结果易于人为计算出，以检查模型的编程是否正确



## To be Faster

-   ```python
    ___ for _ in _ if __ is not x  # ❌
    ```

    ```python
    ___ for _ in _
    __ [x] = __  # ✔️
    ```

-   