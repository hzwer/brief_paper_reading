## Video
### 2023 - [A Dynamic Multi-Scale Voxel Flow Network for Video Prediction](https://arxiv.org/abs/2303.09875)
  * 视频预测，OPT 提出在 Vimeo90K 训练集训练出的插帧模型都能当合理的 loss 度量，跨数据集也泛化得不错，那直接在 Vimeo90K 上训练外插模型应该性能也会很好
  * 整合一些插帧算法的光流估计设计思路，多尺度多阶段地去把目标帧到输入帧的光流估计出来，然后用光流 warping 和遮挡 mask 去混合输入帧，loss 方面用 deep supervision 做一下
  * 有的帧可能有高速行驶的车辆，有的帧可能就是静态的场景，用相同的结构一起做显然是浪费，于是设计出一个能自动跳过一些 block 的网络
  * 经过动态设计，模型推理又能无痛省一半计算量
### 2022 - [Disentangling Architecture and Training for Optical Flow](https://arxiv.org/pdf/2203.10712.pdf)
  * 研究三个经典网络光流估计的结构 ：PWC-Net, IRR-PWC 和 RAFT，在现代的训练技巧下，各种操作带来的提点
  * 只要参数好好调，PWC-Net 的 loss 可以下降到原来的一半（不比 RAFT 差多少）；然后把发现的这些训练 trick 灌给 RAFT，能获得新的 SOTA
  * 宣称PWC-Net 显存开销小，适合做 4k 光流，而且小运动可能做的好点（猜测是 RAFT 8 倍下采样导致）
  * 感觉重要的只有把数据集换成 AutoFlow 和调学习率，还有就是暴力，训几倍时间，没观察到过拟合
  * [想法] 光流估计真的很 data-driven，模型结构设计的收益有点差，甚至不如一直训下去
### 2022 - [Unifying Flow, Stereo and Depth Estimation](https://arxiv.org/pdf/2211.05783.pdf)
  * CVPR22-GMFlow 的期刊版本，将 GMFlow 架构从光流扩展到立体匹配和深度估计
  * 考虑到这几个任务具有内在联系，但目前的模型结构各异，体现在 cost volume 光流是 2D，立体匹配是 1D，作者用统一的框架来解决这些问题
  * GMFlow 的核心思想是用 cross-attention 做特征全局匹配，然后用 softmax 导出光流
  * 立体匹配和深度估计的原理是类似的，基本上是把光流的做法降一个维度
  * 1. 特征提取：用 ResNet 提取出 1/8 分辨率特征，加上 positional encoding。然后把两张图出的 feature 做 cross-attention，这里用的 transformer 是类似 swin 的版本；2. 传播：在特征上直接用 softmax 导出光流显然是不够的，考虑到很多区域可能无法匹配上（因为遮挡等原因），所以求得光流结果以后，再做一次 self-attention 自传播；3. 细化：此前都是在 1/8 分辨率上做计算，为了进一步精细化光流，又把全图分成若干块，每一块在 1/4 分辨率上再计算一遍，两次计算可以 share 一部分的模块参数，具体不表，这一部分称为 “Hierarchical Matching Refinement”
  * 这种架构一个很有意思的好处是，当把 softmax 的维度转一下，就可以从计算正向光流变成计算反向光流
  * [想法] 把光流估计的地位提高了一把，作者的 codebase 写的非常好，可以很方便地把 transformer 相关的代码一键用到其他 model 里
### 2022 - [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/pdf/2011.06294.pdf)
  * 重新审视了以往的光流插帧中的流估计，以及提出一种特权蒸馏的方式帮助模型学习光流
  * 光流网络抛弃预训练，应该针对任务设计以后 from scratch 训练
  * 光流模型需要 coarse-to-fine
  * 以 DAIN 论文为分界线，DAIN 以前有很多 idea 不错的论文，但数据集小效果就不好
  * 遇事不决就 end-to-end，data-driven，手工设计很可能不如 CNN
### 2022 - [VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution](https://arxiv.org/abs/2206.04647)
  * LIIF（把图片表示成隐式场，用 MLP 可以 query 每个像素值，用来实现任意倍率超分）的后续工作
  * 实现任意倍分辨率任意倍速
  * 训练需要两阶段，从易到难
  * [想法] 相比 LIIF 做法没有特别出乎意料，动态场景还是绕不开光流那一套：要用光流，肯定要上数据集，那么就得是个可泛化的场，然后就会越搞越像 CNN
### 2022 - [SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image](https://arxiv.org/pdf/2204.00928.pdf)
  * 单图出 NeRF
  * 约束 NeRF 进行的推算符合已有 depth，然后再用 dino 和 GAN 来约束生成图的纹理和参考图像
  * 这个 GAN 有些神奇，采用了 DiffAugment 做 GAN 的数据增强，DiffAugment 号称一百张图能训练一个 GAN
  * 本文用半监督的世界观将这两个部分描述为 Geometry Pseudo Label 和 Semantic Pseudo Label
  * 对于处于一些遮挡区域的pixel，其对应的depth无法计算确定。为了降低不确定区域的depth uncertainty，论文采用了DDVO 论文中depth smooth约束，即利用pixel的二阶导保证depth的一致性，并且作者还设计了把 depth 映到 reference view 的一致性
  * GAN 的提升有限，cls（局部 texture 监督）还是起到作用了
  * [想法] 有 depth 的情况下实际上转相机是有点 trivial 的，加一些 inpainting 感觉也能出个效果，那么这篇我理解就是把 inpaining 蒸馏进 NeRF 的参数里
  * [想法] 感觉这两年相关领域的重要 trick 都被吸收了
### 2020 - [UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning](https://arxiv.org/pdf/2012.00212.pdf)
  * 无监督光流，trick 大礼包
  * bottom-up problem: 光流上采样时，采用 bilinear/bicubic resize 导致模糊。本文引入了 self-guided upsampling module (SGU)
  * top-down problem: 金字塔网络的中间层缺乏监督。本文引入了 pyramid distillation loss (PDL)
  * [想法] EPE 高的模型在真实视频上可能 warping 结果差，一方面可能是来自于合成数据和真实数据的差异，还可能因为光流定义和指标的缺陷
    举例来说：
     a. 人在运动的时候，按光流的假设，头发的光流和头皮的光流应该一致，然而头发可能有更细微的运动，在插帧的时候我们就希望捕捉到这种细节
     b. 一个纹理非常平坦的物体，在前后帧中，整个物体的所有像素其实都能互相对应，但是光流得学出平滑性
### 2020 - [Softmax Splatting for Video Frame Interpolation](https://arxiv.org/abs/2003.05534)
  * 插帧，主要是提出了 softmax splatting，实现一种 forward-warp
  * 在 forward-warp 时，处理多个像素映射到一个像素点的方法，直接相加会溢出，可以让网络预测一种加权平均，文中对比发现 softmax 比较好
  * 很大部分涨点其实来自于warp context（用光流去 warp pyramid feature map），把它们加入到最后的 U-net 里
  * 有大量 trick，比如 laplacian loss，U-net 的改版 Grid-net 等
### 2020 - [MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://arxiv.org/pdf/2003.10955.pdf)
  * pwc-net 中，使用两个 warped feature 计算 cost volume
  * 从消融实验看，主要是 Asym-conv 涨的点，可能说明 warped feature 与一般 feature 直接进行匹配是不合适的 （作者称：Intuitively, the warping operation induces
ambiguity to the occluded areas and breaks the symmetricity of the feature matching process, so an asymmetric design might be helpful to conquer this divergence.）
### 2020 - [ScopeFlow: Dynamic Scene Scoping for Optical Flow](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bar-Haim_ScopeFlow_Dynamic_Scene_Scoping_for_Optical_Flow_CVPR_2020_paper.pdf)
  * 有监督光流，点很高
  * 从训练集中裁出小图训练，中间的像素被选出的概率大，而边上的像素被选出的概率小，这种偏差对图片光流训练会有很大影响
  * 统计发现，在几个光流训练集中，运动比较小的物体通常在图片中间（远景），而运动大的物体集中在图片边缘
### 2020 - [What Matters in Unsupervised Optical Flow ](https://arxiv.org/pdf/2006.04902.pdf)
  * 通过大量实验分析无监督光流训练中各种方法和技巧的收益，主要包括遮挡处理，图像 loss 设计以及结果的平滑正则
  * 提出了四个改进，包括归一化 costvolumn，遮挡估计时中断梯度，在resize前的光流上加平滑正则，以及自监督训练
  * smoothness，本文推荐使用 edge-aware，鼓励光流和图片有比较一致的边缘
  * 自监督，先对一个 pair 预测光流（teacher），再在图片上加增广预测光流（student），在两个光流之间加自监督 loss，loss 只向 student flow 上传播
### 2020 - [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://discourse.brainpp.cn/t/topic/50779)
  * 光流 best paper
  * 代码比较难懂的就是 core/corr.py 里面，用 grid_sample 实现 lookup 的部分，解读线索是时刻想着对于 I0 的一个像素点 (x,y)，要去 I1 中找 (x + Fx, y + Fy) 的邻域，想办法把它变成一个查表
  * 实验都非常标准，亮点是训练快，且只需要 epe loss 就基本足够了
  * 在 RAFT 的结构设计下，tied weights 比 untied weights 点高，我感觉和作者用了 convGRU 是有关的，通常卷积网络加参数都不太亏。multi scale 和 context 都发挥了挺大作用。lookup radius = 1 掉点不多
  * [想法] RAFT 抛弃了前几年光流估计网络中各种莫名其妙的 trick，但在性能上实现了大幅超越
  * [想法] 据说 RAFT 特点是在 training 时拟合的特别快（因此需要加强增广来提高泛化）：RAFT 看起来工程 trick 少，augmentation 堆的却不少，这部分据说贡献了一半的涨点
  * [想法] 我个人感觉，以往的 warp + cost volume 做法没有这种 lookup 直接，然后 lookup 又能通过多级查找来起一种在 dilated feature 上合并查询一般的效果，所以才涨的点
  * [想法] 近来人们发现光流算法比点大部分就是在比谁遮挡区域解的好，其实是在 overfit 歧义区域吗？
### 2019 - [Depth-Aware Video Frame Interpolation](https://discourse.brainpp.cn/t/topic/18874/2)
  * 在用光流图插帧的时候，考虑图片的深度，即显式地建模遮挡
  * 把 hierarchical features 整合进 Frame Synthesis 网络来提升性能，同时期很多人这么干
  * 训练集是 vimeo90k，训练的时候只预测 t = 0.5，测试时在任意 t 上测试，40 epoch，单卡训了 5 天，参数量是 24M，runtime 为 0.13s（640 × 480)
  * 最终效果很好，但是计算量大，还需要 pretrain 的光流和深度估计
  * [想法] 主表中 runtime 没有在相同硬件报告，RIFE 被坑了一把
### 2016 - [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/pdf/1612.01925.pdf)
  * 使 CNN 光流方法达到了传统光流法的效果
  * 喂不同难度数据的顺序影响模型训练效果
  * 先出一次光流，用光流把 img2 向 img1 warp，然后再把两张图放到下一个网络，级联一些小网络，逐步出残差 refine 之前的光流可以涨点，现在这都是标准操作了
  * 对于小范围的位移构造了一个数据集 ChairsSDHom，并针对改进新的网络结构
  * FlyingChairs, 22k 张图，背景是512*384的flickr图，前景是809种62视角的3d椅子，椅子只有2d移动
  * FlyingThings3D，22k 张图，考虑光线，3d移动，各种物体
  * 直接在FlyingThings3D训不好，先在FlyingChairs上训，再在FlyingThings3D上finetune比较好
  * 尝试用得到的光流来辅助 motion segmentation 和 action recognition 的模型