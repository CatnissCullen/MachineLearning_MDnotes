# Coding Tips

****



## To Debug

-   用较小的值（≈0）或别的特殊值初始化模型参数使得结果易于人为计算出，以检查模型的编程是否正确
-   

## To be Faster

-   ```python
    ___ for _ in _ if __ is not x  # ❌
    ```

    ```python
    ___ for _ in _
    __ [x] = __  # ✔️
    ```

-   