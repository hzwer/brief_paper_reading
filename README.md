# brief_paper_reading
Organize some of my insights and paper reading records. Total Count：20
## LLM
* 2024 - [ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search](https://arxiv.org/pdf/2310.13227.pdf)
  * LLM 的 A*。A* 每次是根据 g(n) 和 h(n) 来选路线的，不需要等模型执行完全过程；在 a* 算法中， 通常我们也会将距离称为代价f，和起点的距离称为历史代价g，和终点的距离称为未来预期代价h ，f=g+h 。距离最近也就是代价最小，就是（g+h）最小。
  * 整体框架是在维护一个搜索树，每次选一个最有前途的叶节点开始扩展，所以这里要把 A* 理解成一种可扩展的广度搜索算法（和算法竞赛用的 A* 不一定一样）
  * g(n) 包括 g1 和 g2，g 的每一个小项相当于选择一个步骤的开销，g1 项是在历史的成功案例中，找一个 lcs_score 最大的，g2 项目是选每个 step 的时候，看一下候选的 k 个 step，有多少和它相似（感觉是 ensemble 变种）
  * h(n) 也是包括 h1 和 h2，前者是把下一个 step 去 memory 中找对照，后者是让 LLM 想象未来还需要具体的多少个步骤
  * 总体计算开销目测是 10x 原始模型，比 mcts 快几倍
* 2024 - [xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval](https://arxiv.org/pdf/2303.03004.pdf)
  * coding 数据集工作，收集了 codeforces 的 7.5k 题目和相关的 5M+ 解答，构造七个任务（检索，翻译等），分析 chatGPT 的表现
  * 数据集的特点是量大、题多、语言多、有部分测例
  * codeforces 比 humaneval 难很多
  * chatGPT 在 codeforces 1600 分段有 10% 的通过率，而我的常识中 chatGPT 很难做对 1200 分及以上的题。作者发现 chatGPT 对于某个时间点后的题正确率陡降，说明应该是背过题库 
* 2023 - [Offline RL for Natural Language Generation with Implicit Language Q Learning](https://openreview.net/pdf?id=aBH_DydEvoH)
  * 强化学习偏好对齐，研究如何用离线 IQL 来训练 NLP 模型，可以看作 reward model + PPO 的一种离线替代；简单来说，Q 函数就是 reward model 的一种扩展，学出 Q 函数再把它加入 inference 就得到了一种带偏好的生成
  * 最原始的 Q 函数回归过程中，需要对每个 state 找一个 Q 最大的 action，在离线学习的过程中，我们只采样了有限的 action，得到对应的 Q，可以看作对于随机变量 x 的若干次采样 si
  * 如果有个网络直接拟合 si，相当于回归平均期望。IQL 套了一个期望回归的方法，使得网络能够回归最大值的期望（具体是通过改 risk function 的角度入手的）。这样就能直接放进原始的离线 DQN 中
  * 除了把 IQL 的想法自然地套到 NLP 任务上，作者还在训练的时候结合了监督学习
  * awac loss 就是让 Q 大的那些 action 提高一下出现频率，CQL loss 是一个正则化
  * [想法] 训练一个 reward model 难度应该不高，ILQL 训练的网络包含了 reward model 的能力后，CQL loss 几乎就是一个变种的监督学习，那么它的下限不应该低于监督学习。上限有多高，能不能达到 instructGPT 中展示的 PPO 的效果有待考察
* 2021 - [Aligning AI With Shared Human Values](https://arxiv.org/pdf/2008.02275.pdf)
  * 构建了一个伦理相关数据集，commonsense, deontology, justice, utilitarianism, and virtue. 即常识、义务论、正义、功利主义和美德。在这些定义下，可以构建争议较小的道德场景判断
  * 题外话：MMLU 评测集中的道德情景，GPT4 表现相当好
  * 作者考虑的道德场景是一般情形下的，举例：闯入另一个人的住所可能是不对的，但是消防员在特殊情形下可以这样做
  * 作者采用众包的形式来对数据进行标注，确保多个人给出一致意见，否则数据会被丢弃。标注人员主要来自英美，作者也评估了用10个印度标注员小规模重标，一致性达到 93% 以上，其中错误的部分还可能来自对习语的误解
  * 正义包含两个部分，一个是 Impartiality（公正），一个是 Desert（应得的）：前者是说一个人所受到的对待不应该由于偏见和歧视被改变，后者是说一个人得到的对待和他的行为是相对对等的
  * [想法] 我在为每个子集编写测试 prompt 的过程中，用 gpt4 来测试可以得到很好的反馈，帮助把 prompt 写的更清晰和消歧义
  * [想法] 大模型自然表现出比小模型更高的道德水平，也有一些文献表明大模型具有自主降低输出毒性的能力
## Base Model
* 2021 - [Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_Diverse_Branch_Block_Building_a_Convolution_as_an_Inception-Like_Unit_CVPR_2021_paper.pdf)
  * RepVGG 后又一篇白嫖涨点的 paper，训练的时候 inception，推理的时候变成 conv 或 resnet
  * 任意加一个带 1x1 的分支，ImageNet 基本上就能涨 0.5+ 的点
  * 按 DBB 的说法，重参数化的关键在于不同分支都要各自带一个 BN
* 2021 - [RepVGG: Making VGG-Style ConvNets Great Again](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)
  * 把 ACNet 的故事上升了一个高度，提出了结构重参数化的概念：“we propose to decouple the training-time multi-branch and inference-time plain architecture via structural re-parameterization”
  * 技术上一言蔽之，把 resblock 通过重参数化改造后变回 3x3 conv，相当把 shortcut 放到了重参数化里，这样可以让 VGG-like 的结构达到 ResNet 的性能
  * 关于 ResNet 好的一个注解："An explanation is that a multi-branch topology, e.g., ResNet, makes the model an implicit ensemble of numerous shallower models, so that training a multi-branch model avoids the gradient vanishing problem"
* 2020 - [Designing network design spaces](https://arxiv.org/pdf/2003.13678.pdf)
  * Ilija 的 RegNet，一种新的模型设计范式，即设计一个好的搜索空间，在里面随机采出的一簇模型平均性能都很好
  * 不断缩小设计空间，使得该空间内模型的平均性能提升，测试方法是在一个空间采 500 个模型，每个模型训 10 epoch
  * 设计目标：简化设计空间结构；提高设计空间的可解释性；改善或维持设计空间的质量；保持设计空间的模型多样性
  * 模型速度跟 (根号 flops) 或者 activation 是线性关系，flops 很容易骗人
  * [想法] 别的很多新文章，本质上想涨点，四个操作，1.加se；2.relu改成prelu或者swish等激活函数；3.加上多尺度信息；4.各种特殊数据扩增，以及更多的epoch，所以我喜欢这篇
* 2019 - [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/pdf/1908.03930.pdf)
  * 重参数化 Rep 宇宙起点（当年大家未发觉）
  * 提出了 不对称的训练 - 推理 方法，实现了推理时免费涨点
  * 作者认为 ACNet 加强了 kernel 骨架的特征提取能力（我觉得是一个简单包装）
* 2016 - [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
  * 这篇是 ResNeXt。AlexNet 曾经把网络分成两组，一组倾向于学习黑白的信息，而另一组倾向于学习到彩色的信息
  * 关于分组，论文说：Multi-head attention allows the model to jointly attend to information from different representation subspaces.
  * 对比 inception 和 ResNeXt，可以看到 ResNeXt 的分支是同构的
## Video
* 2020 - [UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning](https://arxiv.org/pdf/2012.00212.pdf)
  * 无监督光流，trick 大礼包
  * bottom-up problem: 光流上采样时，采用 bilinear/bicubic resize 导致模糊。本文引入了 self-guided upsampling module (SGU)
  * top-down problem: 金字塔网络的中间层缺乏监督。本文引入了 pyramid distillation loss (PDL)
  * [想法] EPE 高的模型在真实视频上可能 warping 结果差，一方面可能是来自于合成数据和真实数据的差异，还可能因为光流定义和指标的缺陷
    举例来说：
     a. 人在运动的时候，按光流的假设，头发的光流和头皮的光流应该一致，然而头发可能有更细微的运动，在插帧的时候我们就希望捕捉到这种细节
     b. 一个纹理非常平坦的物体，在前后帧中，整个物体的所有像素其实都能互相对应，但是光流得学出平滑性
* 2020 - [Softmax Splatting for Video Frame Interpolation](https://arxiv.org/abs/2003.05534)
  * 插帧，主要是提出了 softmax splatting，实现一种 forward-warp
  * 在 forward-warp 时，处理多个像素映射到一个像素点的方法，直接相加会溢出，可以让网络预测一种加权平均，文中对比发现 softmax 比较好
  * 很大部分涨点其实来自于warp context（用光流去 warp pyramid feature map），把它们加入到最后的 U-net 里
  * 有大量 trick，比如 laplacian loss，U-net 的改版 Grid-net 等
* 2020 - [MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://arxiv.org/pdf/2003.10955.pdf)
  * pwc-net 中，使用两个 warped feature 计算 cost volume
  * 从消融实验看，主要是 Asym-conv 涨的点，可能说明 warped feature 与一般 feature 直接进行匹配是不合适的 （作者称：Intuitively, the warping operation induces
ambiguity to the occluded areas and breaks the symmetricity of the feature matching process, so an asymmetric design might be helpful to conquer this divergence.）
* 2020 - [ScopeFlow: Dynamic Scene Scoping for Optical Flow](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bar-Haim_ScopeFlow_Dynamic_Scene_Scoping_for_Optical_Flow_CVPR_2020_paper.pdf)
  * 有监督光流，点很高
  * 从训练集中裁出小图训练，中间的像素被选出的概率大，而边上的像素被选出的概率小，这种偏差对图片光流训练会有很大影响
  * 统计发现，在几个光流训练集中，运动比较小的物体通常在图片中间（远景），而运动大的物体集中在图片边缘
* 2020 - [What Matters in Unsupervised Optical Flow ](https://arxiv.org/pdf/2006.04902.pdf)
  * 通过大量实验分析无监督光流训练中各种方法和技巧的收益，主要包括遮挡处理，图像 loss 设计以及结果的平滑正则
  * 提出了四个改进，包括归一化 costvolumn，遮挡估计时中断梯度，在resize前的光流上加平滑正则，以及自监督训练
  * smoothness，本文推荐使用 edge-aware，鼓励光流和图片有比较一致的边缘
  * 自监督，先对一个 pair 预测光流（teacher），再在图片上加增广预测光流（student），在两个光流之间加自监督 loss，loss 只向 student flow 上传播
* 2019 - [Depth-Aware Video Frame Interpolation ](https://discourse.brainpp.cn/t/topic/18874/2)
  * 在用光流图插帧的时候，考虑图片的深度，即显式地建模遮挡
  * 把 hierarchical features 整合进 Frame Synthesis 网络来提升性能，同时期很多人这么干
  * 训练集是 vimeo90k，训练的时候只预测 t = 0.5，测试时在任意 t 上测试，40 epoch，单卡训了 5 天，参数量是 24M，runtime 为 0.13s（640 × 480)
  * 最终效果很好，但是计算量大，还需要 pretrain 的光流和深度估计
  * [想法] 主表中 runtime 没有在相同硬件报告，RIFE 被坑了一把
* 2016 - [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/pdf/1612.01925.pdf)
  * 使 CNN 光流方法达到了传统光流法的效果
  * 喂不同难度数据的顺序影响模型训练效果
  * 先出一次光流，用光流把 img2 向 img1 warp，然后再把两张图放到下一个网络，级联一些小网络，逐步出残差 refine 之前的光流可以涨点，现在这都是标准操作了
  * 对于小范围的位移构造了一个数据集 ChairsSDHom，并针对改进新的网络结构
  * FlyingChairs, 22k 张图，背景是512*384的flickr图，前景是809种62视角的3d椅子，椅子只有2d移动
  * FlyingThings3D，22k 张图，考虑光线，3d移动，各种物体
  * 直接在FlyingThings3D训不好，先在FlyingChairs上训，再在FlyingThings3D上finetune比较好
  * 尝试用得到的光流来辅助 motion segmentation 和 action recognition 的模型
## Low-Level Vision 
  * 2022 - [Simple Baselines for Image Restoration ](https://arxiv.org/abs/2204.04676)
    * 提出一个图像修复的简单基线模型，核心是带 layernorm 的深层模型和本文提出的非线性无激活组件（用乘法代替激活函数）
    * NAFNet 的核心是 layernorm 和 simplegate (Gate(X, f, g, σ) = f(X) ⊙ σ(g(X)))
    * low-level vision 做到 layernorm + 深这两点，性能就可以很好
  * 2022 - [RepSR: Training Efficient VGG-style Super-Resolution Networks with Structural Re-Parameterization and Batch Normalization](https://dl.acm.org/doi/abs/10.1145/3503161.3547915)
    * BN 可以帮助超分模型训练，但因为训练推理的不对称性，会出现一些 artifacts
    * 把重参数化放到 SR 里，很多类似 paper
    * 在超分上，BN 的统计量有点问题，观察训练曲线，发现 validation 时常爆炸，train curve 一直很正常
    * 在训练最后阶段用 population 统计量代替 mini-batch 统计量（我理解就是把 BN 切换成 eval 模式再微调），涨了一些点   
## Reinforcement Learning
* 2023 - [Learning About Progress From Experts](https://openreview.net/pdf?id=sKc6fgce1zs)
  * 强化学习，nethack 这类流程步骤非常长的游戏，直接从显式奖励中并不好学习，本文提出专家的示例隐含着对游戏进程推进的指示信息，可以先从专家的游戏视频中学出一个指示游戏进程的 progress model，来提供 reward
  * 从专家数据集里抽出同 episode 的两帧，让 progress model 估计这两帧之间的有向时间距离，然后把这一项加到原始 reward 里
  * 一个细节是需要把 state 里面直接表示时间的部分手动移除掉
  * process model 可以自动把揭示地图迷雾和游戏进程的推进联系起来，也能把注意到各种属性的意义
  * 训练 progress model 时，随机选取专家轨迹中的两帧，让 model 估计这两帧的有向时间差；在训练 agent 的时候，选择让 progress model 衡量当前 t 时刻局面和 t-8 时刻局面的进程差，作为一种奖励
  * 用 learning 的方式替代对于各种属性特征的手工奖励建模，在 nethack 中非常合理
