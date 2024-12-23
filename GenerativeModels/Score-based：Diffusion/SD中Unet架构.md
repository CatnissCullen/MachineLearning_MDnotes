# Stable Diffusion 中的 Unet 架构

****



## 基础块

<img src="./img/DownSample.png" alt="DownSample" style="zoom: 33%;" />

<img src="./img/UpSample.png" alt="UpSample" style="zoom: 33%;" />

<img src="./img/image-20240630141859024.png" alt="image-20240630141859024" style="zoom: 33%;" />

<img src="./img/ResBlock.png" alt="ResBlock" style="zoom: 33%;" />

<img src="./img/AttnBlock.png" alt="AttnBlock" style="zoom:33%;" />

<img src="./img/FeedForward.png" alt="FeedForward" style="zoom:33%;" />



## 组合块

![BasicTransformerBlock](./img/BasicTransformerBlock.png)

<u>***BasicTransformerBlock 即 Basic ViT Block***</u> 

![SpatialTransformer](./img/SpatialTransformer.png)

<u>***BasicTransformerBlock 可叠加多次***</u> 

![Conditioned_DownBlock](./img/Conditioned_DownBlock.png)

<u>***也可以用简单的 Attention 而不用 Spatial Transformer***</u>  

<img src="./img/DownBlock.png" alt="DownBlock" style="zoom:33%;" />

<img src="./img/Conditioned_MiddleBlock.png" alt="Conditioned_MiddleBlock" style="zoom:33%;" />

<img src="./img/UpBlock.png" alt="UpBlock" style="zoom:33%;" />

<img src="./img/Conditioned_Upblock.png" alt="Conditioned_Upblock" style="zoom:33%;" />



## 整体架构

代码路径：**`LDM/ldm/modules/diffusionmodules/openaimodel.py`** 

代码概述见：[S:\CAMPUS\AI\MyProject\DiffusionPainter\LDM\scripts\README.md](S:\CAMPUS\AI\MyProject\DiffusionPainter\LDM\scripts\README.md) 

代码解读见：[https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html ](https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html) 

![SD_Unet](./img/SD_Unet.png)

![image-20240918160852126](./img/image-20240918160852126.png)

>   **the U-Net has three main stages: down-stage, mid-stage, and up-stage.** 
>
>   -   The down-stage reduces the resolution of activations, 
>   -   while the up-stage increases it. 
>
>   Both stages contain multiple resolutions, in each of which the activations share the same resolution. Furthermore, <u>each resolution includes several modules, including **ResModule** (convolutional ResNet structures), **ViT Module**, and **Downsampler/Upsampler** (simple convolutional layers)</u>.  
>
>   *--- 《Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features》*