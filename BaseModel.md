## Base Model
### 2022 - [Online convolutional re-parameterization](http://openaccess.thecvf.com/content/CVPR2022/html/Hu_Online_Convolutional_Re-Parameterization_CVPR_2022_paper.html)
  * 这篇一个核心的证明是说，带有 scaling 的多分支卷积不会退化成单个卷积
  * 这样就不强求重参数化要用 BN 了
### 2021 - [Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_Diverse_Branch_Block_Building_a_Convolution_as_an_Inception-Like_Unit_CVPR_2021_paper.pdf)
  * RepVGG 后又一篇白嫖涨点的 paper，训练的时候 inception，推理的时候变成 conv 或 resnet
  * 任意加一个带 1x1 的分支，ImageNet 基本上就能涨 0.5+ 的点
  * 按 DBB 的说法，重参数化的关键在于不同分支都要各自带一个 BN
### 2021 - [RepVGG: Making VGG-Style ConvNets Great Again](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)
  * 把 ACNet 的故事上升了一个高度，提出了结构重参数化的概念：“we propose to decouple the training-time multi-branch and inference-time plain architecture via structural re-parameterization”
  * 技术上一言蔽之，把 resblock 通过重参数化改造后变回 3x3 conv，相当把 shortcut 放到了重参数化里，这样可以让 VGG-like 的结构达到 ResNet 的性能
  * 关于 ResNet 好的一个注解："An explanation is that a multi-branch topology, e.g., ResNet, makes the model an implicit ensemble of numerous shallower models, so that training a multi-branch model avoids the gradient vanishing problem"
### 2020 - [Designing network design spaces](https://arxiv.org/pdf/2003.13678.pdf)
  * Ilija 的 RegNet，一种新的模型设计范式，即设计一个好的搜索空间，在里面随机采出的一簇模型平均性能都很好
  * 不断缩小设计空间，使得该空间内模型的平均性能提升，测试方法是在一个空间采 500 个模型，每个模型训 10 epoch
  * 设计目标：简化设计空间结构；提高设计空间的可解释性；改善或维持设计空间的质量；保持设计空间的模型多样性
  * 模型速度跟 (根号 flops) 或者 activation 是线性关系，flops 很容易骗人
  * [想法] 别的很多新文章，本质上想涨点，四个操作，1.加se；2.relu改成prelu或者swish等激活函数；3.加上多尺度信息；4.各种特殊数据扩增，以及更多的epoch，所以我喜欢这篇
### 2019 - [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/pdf/1908.03930.pdf)
  * 重参数化 Rep 宇宙起点（当年大家未发觉）
  * 提出了 不对称的训练 - 推理 方法，实现了推理时免费涨点
  * 作者认为 ACNet 加强了 kernel 骨架的特征提取能力（我觉得是一个简单包装）
### 2016 - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
  * 这篇是 ResNeXt。AlexNet 曾经把网络分成两组，一组倾向于学习黑白的信息，而另一组倾向于学习到彩色的信息
  * 关于分组，论文说：Multi-head attention allows the model to jointly attend to information from different representation subspaces.
  * 对比 inception 和 ResNeXt，可以看到 ResNeXt 的分支是同构的
