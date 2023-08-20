# Coding Tips

****



## For Convenience

-   把偏差 $b$ 看作属性值为常数1的一项的权重，合并到权重矩阵的最右一列，并且给输入向量（属性值向量）多增加恒为一的一元

    <img src="images/image-20230815102933855.png" alt="image-20230815102933855" style="zoom: 33%;" />
    
    ```python
    # append the bias dimension of ones (i.e. bias trick) so that our model only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    ```
    
    

## For Prcision

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