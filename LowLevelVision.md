## Low-Level Vision
### 2022 - [Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/abs/2112.01455)
  * 本文训练了一个 NeRF + CLIP，根据输入的自然语言生成 3d 辐射场，主要技术点是研究怎么对生成过程进行合理约束。
  * 模型用的是一个简化版的 NeRF，输入一个 position x，多层 MLP 输出 density 和 color（漫反射），训练中每次随机一个 view，渲染一张图片，损失函数主要是由预训练的 CLIP 模型来给出 text 和某个视角下生成结果的距离。
  * 每次把图片喂给 CLIP 前，加一个随机的背景。作者设计了三种背景生成方式。去掉这个 trick 以后，生成的体素倾向于散开。
作者还显式约束了 mean density，加了一项比较简单的 loss。
  * 100K iterations 需要 8TPU 跑 72min，但跑 4K 的时候已经不错效果了
  * 注意到一个 CLIP 模型（比如 B/32) 指导的生成过程，会对这个 CLIP 过拟合，换一个 CLIP 点就差别很大
  * 不借助预训练的生成模型，直接优化 CLIP loss 就能做生成，近期有很多类似工作
### 2022 - [Simple Baselines for Image Restoration ](https://arxiv.org/abs/2204.04676)
  * 提出一个图像修复的简单基线模型，核心是带 layernorm 的深层模型和本文提出的非线性无激活组件（用乘法代替激活函数）
  * NAFNet 的核心是 layernorm 和 simplegate (Gate(X, f, g, σ) = f(X) ⊙ σ(g(X)))
  * low-level vision 做到 layernorm + 深这两点，性能就可以很好
### 2022 - [RepSR: Training Efficient VGG-style Super-Resolution Networks with Structural Re-Parameterization and Batch Normalization](https://dl.acm.org/doi/abs/10.1145/3503161.3547915)
  * BN 可以帮助超分模型训练，但因为训练推理的不对称性，会出现一些 artifacts
  * 把重参数化放到 SR 里，很多类似 paper
  * 在超分上，BN 的统计量有点问题，观察训练曲线，发现 validation 时常爆炸，train curve 一直很正常
  * 在训练最后阶段用 population 统计量代替 mini-batch 统计量（我理解就是把 BN 切换成 eval 模式再微调），涨了一些点
### 2020 - [Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence](https://arxiv.org/pdf/2005.05207.pdf)
  * sketch 上色的一个流派，reference-based
  * 一张图不知道怎么找参考图的时候，就把自身做一些增广以后当参考图
  * 本文设计的 spatially corresponding feature transfer module 可能能用于其它任务
### 2019 - [Single Image Reﬂection Removal Exploiting Misaligned Training Data and Network Enhancements](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Single_Image_Reflection_Removal_Exploiting_Misaligned_Training_Data_and_Network_CVPR_2019_paper.pdf)
  * 一篇做图片去反射的问题，主要提高了模型在真实数据上的表现
  * 亮点主要在损失函数部分，提出了对于不对齐的训练数据计算损失函数的一些方法
  * 作者认为 vgg 由于存在大量的 maxpooling，对于不对齐应该是不敏感的，于是分析了一下能否用某一层的 feature 来计算 loss
  * 用 conv5_2 计算 loss，训练结果接近 contextual loss（这个 loss 的计算量巨大）
  * [想法] misalign 的问题我思考了好久，没想到用 perceptual loss 就能解的这么好
### 2019 - [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/pdf/1903.09291.pdf)
  * 一篇比较早的 CNN 剪枝文章，提出 GAL（Generative Adversarial Learning）；用 softmask 来剪去一些 CNN 分支，用判别器来计算剪枝前后 feature 上的 loss；这个操作在 MEALv2 后被大家熟知
  * 基于GAL的剪枝策略，能够在对抗学习过程中避免标注的使用；其次，Soft Pruning Mask 使正则化过程变得更加松弛、更容易学习收敛；另外，对抗训练与正则化过程是端到端的、非逐层的（作者认为这是一个优点）。
  * 在对抗学习过程中，Baseline的参数固定，而剪枝模型参数、soft mask以及判别器参数在训练中更新。
  * 训练过程主要包含两个交替的阶段：第一个阶段固定生成器和mask，通过对抗训练更新判别器D，损失函数包含对抗损失与对抗正则项；第二阶段固定判决器，更新生成器与mask，损失函数包含对抗损失中的生成器与baseline特征输出的MSE损失以及生成器和mask的正则项。最终，根据mask的阈值和门控方式，对channel、branch或block进行剪枝，从而实现模型的压缩。