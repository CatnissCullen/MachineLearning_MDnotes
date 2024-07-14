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

![SpatialTransformer](./img/SpatialTransformer.png)

![Conditioned_DownBlock](./img/Conditioned_DownBlock.png)

<img src="./img/DownBlock.png" alt="DownBlock" style="zoom:33%;" />

<img src="./img/Conditioned_MiddleBlock.png" alt="Conditioned_MiddleBlock" style="zoom:33%;" />

<img src="./img/UpBlock.png" alt="UpBlock" style="zoom:33%;" />

<img src="./img/Conditioned_Upblock.png" alt="Conditioned_Upblock" style="zoom:33%;" />



## 整体架构

代码路径：**`LDM/ldm/modules/diffusionmodules/openaimodel.py`** 

代码概述见：[S:\CAMPUS\AI\MyProject\DiffusionPainter\LDM\scripts\README.md](S:\CAMPUS\AI\MyProject\DiffusionPainter\LDM\scripts\README.md) 

代码解读见：[https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html ](https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html) 

![SD_Unet](./img/SD_Unet.png)