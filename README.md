# brief_paper_reading
主要是记录一些这几年读的 paper，持续搬运中，欢迎指正

主要领域 LLM，Video, Low-level Vision, Reinfercement Learning

扩展阅读：
[走出新手村」十次 CV 论文会议投稿的经验总结.pdf](https://drive.google.com/file/d/1w2ZgIF1Q92Li_p7pCDPJFT_0312Cv2QC/view?usp=sharing)

Organize some of my insights and paper reading records. Total Count：52
## LLM
### 2024 - [RealChat-1M: A Large-Scale Real-World LLM Conversation Dataset](https://openreview.net/forum?id=BOfDKxfwt0)
  * 一篇做数据集的工作，包含 1M 轮人类和 LLM 的对话（平均两对），和以往数据集的对比，主要突出一个大
  * 收集这个数据主要的意义：一方面是学术界缺乏大规模的用户 prompt，鼓励用户和开源的弱智模型进行百万次交互本身就很困难；另一方面 vicuna 一定程度说明了收集这样的用户问题加上大模型的回答，可能可以调校出很好的对话模型。
  * 大部分数据是 vicuna13b 生成的（作者平台的默认模型），但作者强调说如果想要获得高质量一些的结果，可以把全数据集用 gpt4 出一遍。
  * 最多的几种语言是英语、葡萄牙语、俄语、汉语和西班牙语
  * 从大类上看，软件、程序相关和内容生成是比较多的请求
  * 用法示例 1 内容审核：虽然 openai 提供了比较准确的内容审核 api，但依然可以人工找到很多漏网之鱼，在用户的毒 instruction 攻击下， llama2-chat 防的比较好（所有毒性检测是由 openai 的 api 完成的）；从每一类中选了 1k 个样本，让 gpt4 生成将其标记为不良内容的原因作蒸馏，并加上了 1k 正常样本和 3k 的 sharegpt 样本进行训练，这样能使得 7B 模型也有较强的判别效果 
  * 用法示例 2 安全基准：作者挑选了对于不同模型攻击成功的50个示例来构建一个攻击 benchmark，该 benchmark 可以用来测试模型的安全性
  * 用法示例 3 SFT：作者抽了两个训练集训练 llama2-7B，HQ是由claude或者chatgpt作回答的子集，Upvote是由用户点赞的数据，Vicuna > LLama > HQ > Upvote，HQ 只有 Vicuna 训练集 的 1/10，依然达到了很不错的性能
  * 用法示例 4 模型评测：作者提供的平台上，支持模型对战操作，作者认为在这个模式下，用户提出的问题可能更有判别性。作者用 gpt3.5 对 50k prompt 进行了打分，分数越高意味着“a greater potential to evaluate the LLMs in problem-solving, creativity and truthfulness”；作者从 >8 和 <2 中各抽了 50 个题，发现在 <2 子集中，确实难以区分 GPT4 和 GPT3.5，而 >8 子集则有较强的区分
  * 除此之外，这个数据集也可能用于隐私、安全、RLHF 等方向的研究
### 2024 - [ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search](https://arxiv.org/pdf/2310.13227.pdf)
  * LLM 的 A*。A* 每次是根据 g(n) 和 h(n) 来选路线的，不需要等模型执行完全过程；在 a* 算法中， 通常我们也会将距离称为代价f，和起点的距离称为历史代价g，和终点的距离称为未来预期代价h ，f=g+h 。距离最近也就是代价最小，就是（g+h）最小。
  * 整体框架是在维护一个搜索树，每次选一个最有前途的叶节点开始扩展，所以这里要把 A* 理解成一种可扩展的广度搜索算法（和算法竞赛用的 A* 不一定一样）
  * g(n) 包括 g1 和 g2，g 的每一个小项相当于选择一个步骤的开销，g1 项是在历史的成功案例中，找一个 lcs_score 最大的，g2 项目是选每个 step 的时候，看一下候选的 k 个 step，有多少和它相似（感觉是 ensemble 变种）
  * h(n) 也是包括 h1 和 h2，前者是把下一个 step 去 memory 中找对照，后者是让 LLM 想象未来还需要具体的多少个步骤
  * 总体计算开销目测是 10x 原始模型，比 mcts 快几倍
### 2024 - [xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval](https://arxiv.org/pdf/2303.03004.pdf)
  * coding 数据集工作，收集了 codeforces 的 7.5k 题目和相关的 5M+ 解答，构造七个任务（检索，翻译等），分析 chatGPT 的表现
  * 数据集的特点是量大、题多、语言多、有部分测例
  * codeforces 比 humaneval 难很多
  * chatGPT 在 codeforces 1600 分段有 10% 的通过率，而我的常识中 chatGPT 很难做对 1200 分及以上的题。作者发现 chatGPT 对于某个时间点后的题正确率陡降，说明应该是背过题库 
### 2023 - [Offline RL for Natural Language Generation with Implicit Language Q Learning](https://openreview.net/pdf?id=aBH_DydEvoH)
  * 强化学习偏好对齐，研究如何用离线 IQL 来训练 NLP 模型，可以看作 reward model + PPO 的一种离线替代；简单来说，Q 函数就是 reward model 的一种扩展，学出 Q 函数再把它加入 inference 就得到了一种带偏好的生成
  * 最原始的 Q 函数回归过程中，需要对每个 state 找一个 Q 最大的 action，在离线学习的过程中，我们只采样了有限的 action，得到对应的 Q，可以看作对于随机变量 x 的若干次采样 si
  * 如果有个网络直接拟合 si，相当于回归平均期望。IQL 套了一个期望回归的方法，使得网络能够回归最大值的期望（具体是通过改 risk function 的角度入手的）。这样就能直接放进原始的离线 DQN 中
  * 除了把 IQL 的想法自然地套到 NLP 任务上，作者还在训练的时候结合了监督学习
  * awac loss 就是让 Q 大的那些 action 提高一下出现频率，CQL loss 是一个正则化
  * [想法] 训练一个 reward model 难度应该不高，ILQL 训练的网络包含了 reward model 的能力后，CQL loss 几乎就是一个变种的监督学习，那么它的下限不应该低于监督学习。上限有多高，能不能达到 instructGPT 中展示的 PPO 的效果有待考察
### 2023 - [MarioGPT: Open-Ended Text2Level Generation through Large Language Models](https://arxiv.org/pdf/2302.05981.pdf)
  * 机器之心： 用 GPT 生成《超级马里奥》游戏关卡，近 9 成关卡可玩
  * MarioGPT 是一个经过微调的 GPT-2 模型（更具体地，是基于一个 86M 大小的 DistilGPT2），经过训练可以生成《超级马里奥》中的游戏关卡
  * 一个关卡可以看成字符画，GPT 每次处理一列
  * MarioGPT 已经可以直接生成关卡，但我们不仅要生成具有不同物理特征的关卡，而且要生成让玩家能觉得有趣的关卡
  * 维护一个关卡集合（起始 30 个），生成新的关卡后拿 A* 算法评估通关轨迹，看看相比集合中的有没有新颖度，生成的方法也是渐进地变异杂交
  * 为了测试可玩性，研究者在 250 个关卡中部署了 Robin Baumgarten 的 A* agent。研究者发现，所有 MarioGPT 生成的关卡中，88.33% 可以由 agent 完成
### 2023 - [A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Instruction Following Large Language Model](https://github.com/LianjiaTech/BELLE/blob/main/docs/A%20Comparative%20Study%20between%20Full-Parameter%20and%20LoRA-based.pdf)
  * belle 团队探讨了全参数 sft 和 lora sft 的效果差异，讨论了训练开销和模型性能之间的取舍
  * 比较显著的差异是学习率，lora sft 可以使用 40x 的学习率
  * [想法] lora 上做研究可能会很快达到瓶颈，最后大家被迫选择全参数
### 2023 - [OctoPack: Instruction Tuning Code Large Language Models](https://arxiv.org/pdf/2308.07124.pdf)
  * 这篇提出了 commit pack 数据集，是 4T 的预训练数据；还提出了 humanevalpack benchmark，包括六种编程语言(Python, JavaScript, Java, Go, C++, Rust)下的写代码、修代码、代码解释任务
  * commit pack 的子集 (74w sample) 做 sft，使得 6B 模型超过之前 starcoder 16B 的效果
  * 作者考虑到了几类任务 HUMANEVALFIX (NL+C→C) ，HUMANEVALEXPLAIN (NL+C→NL) ，HUMANEVALSYNTHESIZE (NL→C) ；其中 Explanation 的评测方式，是让模型解释代码，根据解释重新生成一份代码，再看新代码通过率
  * 主要有三个结论：COMMITPACKFT enables CodeLLMs to fix bugs ：是说 CommitPack 主要在 Code Fixing 帮助涨点；Importance of samples with natural language targets ：只有代码预训练的模型，不会解释代码；COMMITPACKFT+OASST yields best performance
  * 预训练的 bloomz，在 Go 和 Rust 上是 0 分（因为语料相对太少），但是经过 sft 后有一些分数，而 sft 数据中并没有 Go 和 Rust 相关内容，作者因此说 Instruction tuning generalizes to unseen programming languages
  * 作者提出的第二个结论是 Pretraining weight correlates with programming language performance after instruction tuning ，直觉上很自然，某种编程语言在预训练见的越多，sft 后该语言的性能就越好，比如 Rust 占的比重最少 (1.5%)，因此表现就最差
  * 当前挑战：Models struggle with small targeted changes ，是说即使两个任务非常相近，模型也可能会做其中一个而不会做另一个；Models struggle switching between code and text 模型在 code 和 text 之间切换困难；Models struggle adhering to a specified output length 不会数字数
  * 附录中给了 commit 相关过滤词表还有各种统计
### 2022 - [SCIENCEWORLD: Is your Agent Smarter than a 5th Grader?](https://arxiv.org/pdf/2203.07540.pdf)
  * LLM 在检索查表方面很强，但它们并不能在一些非常简单的实验场景验证科学概念，比如构建实验判断某个物体是否导电
  * 本文提出了一组 30 个科学实验相关的 RL tasks，涉及简单的理科实验，LLM 目前尚不能给解决这些任务带来增益
  * 作者分析说由于动作空间有百万种组合，大模型受困于无法找到合法的 action
  * [想法] 将 LLM grounding 到具体 RL 任务上时，目前还没有通用的训练范式
### 2022 - [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068?source=techstories.org)
  * 汇集了很多大家已知的 Transformer 训练问题，而且由于时间短，基建任务重，模型大，踩了很多坑
  * 要在三个多月用千卡 A100 训练一个 175B 的模型，其中完整训练需要 33 天；团队有五个人，还加上一些云服务商支持
  * 硬件故障：35次手动重启，100卡出故障被移除
  * Loss 时常训飞，每当训飞的时候，降低学习率，加载上一个 checkpoint
  * 人肉 PID learning rate
  * 权重初始化和 Megatron-LM 开源的代码中保持一致，使用了均值 0 标准差 0.006 的分布来初始化，输出层的标准差用 1.0/开根号(2L) 来做缩放，L是总共的层数
  * 使用 AdamW 作为优化器，(β1,β2) = (0.9,0.95)，weight decay 是 0.1（默认是 0.01，据说在大模型训练时偏小）
  * 作者发现 gradient clipping 一般在 0.2 左右，所以直接用 clip 0.3 来稳定训练
  * 尝试 SGD，努力以后放弃；用和 adam 相同学习率会很快就不下降了，用更大的学习率也没有明显收益
  * [来自知乎] 自BERT和GPT之后，NLP领域的技术就在原地打转了，BERT在2018年9月出来，大家经过四年的努力，发现Scale up 是目前唯一正确的解决方案 
### 2022 - [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
  * 这篇 paper 研究如何在保持 LM AI 智商的情况下，使得它更无害（减少胡说八道），给出的方案是用自学习的办法
  * harmlessness 和 helpfulness 之间不完全兼容（例如对所有问题都回答不知道的 LM 是无害的，但是也没用了），本文试图提出一种帕累托改进方案
  * 首先作者人工迭代了十条由语言描述的 原则，做法主要有两部分：第一步是监督学习：Critique → Revision → Supervised Learning
  * 简单来说就是让 LM 对一些有害的提示词作回复，这时候 LM 往往也产生有害的内容，接着随机选一些原则，让 LM 对它的回答进行修改，这样相当于产生了修正的标签，用这些标签进行监督学习
  * 第二步是强化学习：AI Comparison Evaluations → Preference Model → Reinforcement Learning
  * 作者说这里类似 RLHF，在 RLHF 中，Human feedback 被蒸馏成 reward model；在这里，由人类来对 LM 回复的 helpfulness 进行打分，而 harmfulness 依然由 AI 自己打分，这样构建数据集得到一个 preference model（PM），最后由 PM + RL 再调整一下原来的 LM 模型
  * 在这个过程中 CoT 可以直接把大模型在 HHH Eval 上的性能提高一截
### 2022 - [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
  * 本文探讨了模型大小 N 和训练数据量 D 的匹配关系，提出现有的大模型都是训练不充分的，当模型大小倍增时，训练数据量也应该倍增，作者最终得到一个效果很好的 70B chinchilla
  * 在给定训练计算资源的情况下，作者做了一系列不同大小的模型实验，并把不同计算资源下的最优的配置记录下来，可以看到，随着计算资源的增长，数据和模型参数量应当以同数量级增长
  * 上述第一个实验是固定一个模型的参数量大小，然后在整个训练过程中观察性能变化
  * 和前一个实验类似，第二个实验对于给定的计算资源，看不同参数量的模型的最终训练效果，结论是类似的
  * 把上面两组实验的所有结果放在一起，通过“classical risk decomposition” 拟合，认为 Gopher 最优参数量在 40B 左右，而当前的 Gopher 参数量是 280B
  * 综上，作者估计当前 Gopher 模型大小需要 17 倍训练时长才充分，或者应当把模型大小调整到 40B-70B
  * [想法] 论文观感相当好，efficient net 既视感
### 2021 - [Aligning AI With Shared Human Values](https://arxiv.org/pdf/2008.02275.pdf)
  * 构建了一个伦理相关数据集，commonsense, deontology, justice, utilitarianism, and virtue. 即常识、义务论、正义、功利主义和美德。在这些定义下，可以构建争议较小的道德场景判断
  * 题外话：MMLU 评测集中的道德情景，GPT4 表现相当好
  * 作者考虑的道德场景是一般情形下的，举例：闯入另一个人的住所可能是不对的，但是消防员在特殊情形下可以这样做
  * 作者采用众包的形式来对数据进行标注，确保多个人给出一致意见，否则数据会被丢弃。标注人员主要来自英美，作者也评估了用10个印度标注员小规模重标，一致性达到 93% 以上，其中错误的部分还可能来自对习语的误解
  * 正义包含两个部分，一个是 Impartiality（公正），一个是 Desert（应得的）：前者是说一个人所受到的对待不应该由于偏见和歧视被改变，后者是说一个人得到的对待和他的行为是相对对等的
  * [想法] 我在为每个子集编写测试 prompt 的过程中，用 gpt4 来测试可以得到很好的反馈，帮助把 prompt 写的更清晰和消歧义
  * [想法] 大模型自然表现出比小模型更高的道德水平，也有一些文献表明大模型具有自主降低输出毒性的能力
 ### 2018 - [BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning](https://arxiv.org/pdf/1810.08272.pdf)
   * 一个研究 Sample Efficiency of Grounded Language Learning 的平台
   * 有从易到难的 19 个任务，并且包括程序写的解法用来克隆学习；模拟很快，每秒也是千帧级，而且可回溯；Goal 用语言描述，虽然不是纯自然语言，但也是一个非常大的组合空间，称为 baby language
   * 在 1M 交互的训练下，模型能有不错的结果，一些预训练对更高级任务是有帮助的
   * 先用一个很小的数据集训 agent，训完以后加入一部分它 fail 的带标注数据，如此反复迭代；像是一种 hard example mining
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
## Reinforcement Learning
### 2023 - [Learning About Progress From Experts](https://openreview.net/pdf?id=sKc6fgce1zs)
  * 强化学习，nethack 这类流程步骤非常长的游戏，直接从显式奖励中并不好学习，本文提出专家的示例隐含着对游戏进程推进的指示信息，可以先从专家的游戏视频中学出一个指示游戏进程的 progress model，来提供 reward
  * 从专家数据集里抽出同 episode 的两帧，让 progress model 估计这两帧之间的有向时间距离，然后把这一项加到原始 reward 里
  * 一个细节是需要把 state 里面直接表示时间的部分手动移除掉
  * process model 可以自动把揭示地图迷雾和游戏进程的推进联系起来，也能把注意到各种属性的意义
  * 训练 progress model 时，随机选取专家轨迹中的两帧，让 model 估计这两帧的有向时间差；在训练 agent 的时候，选择让 progress model 衡量当前 t 时刻局面和 t-8 时刻局面的进程差，作为一种奖励
  * 用 learning 的方式替代对于各种属性特征的手工奖励建模，在 nethack 中非常合理
### 2023 - [Mastering Diverse Domains through World Models](https://danijar.com/project/dreamerv3/)
  * 对 Dreamerv2 进行可扩展性方面的升级， 能够在跨domain无需调整超参数的情况下，可以实现包括连续和离散动作，视觉和低维输入，2D和3D世界，不同的数据预算、奖励频率和奖励等级的全部任务
  * 在各种需要回归数值的地方放上符号对数预测，使得各种定义的尺度变化对模型影响变小
  * 还有一个改进是作者认为 critic 直接回归值不够好，改成回归分布
  * 使模型完全可以离线训练，replayed step 倍数越大，训的越好
  * 大 policy 模型不仅上限高，而且数据效率也高
### 2023 - [Become a Proficient Player with Limited Data through Watching Pure Videos](https://openreview.net/forum?id=Sy-o2N0hF4f)
  * 提供了一种从纯 video 预训练强化学习模型的方法
  * 专门收集数据贵，只看 video 来预训练会更方便，核心就是构建某种无监督 consistency，在 atari 任务上训练
  * 预训练的时候不用真的去拟合动作空间，只要学一个动作隐空间即可，finetune 的时候再学，构建重建和前后一致性的 loss
  * [想法] 感觉还挺靠谱的，技术上的贡献比较扎实
### 2022 - [NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning](https://arxiv.org/pdf/2102.00714.pdf)
  * 本文收集了一些 benchmark，做了比较大规模的 offline RL 算法评测，结论是在 online evaluation（模拟场景中测试和训练）中，只有一个 CQL 算法比 naive baseline 好，而用来现实部署时，性能基本是随机的
  * 作者认为现有框架中存在的问题：1. Conservative data：因为采集成本的原因，在数据采集时，操作通常是保守的 2.Limited available data：有的时候，用户数据就是很难大量拿到 3.Highly stochastic environments 现实世界固有的 stochastic nature 4. Offline evaluation before deployment：说白了很多 RL benchmark 就是在 overfit
  * 在 Online Model Selection 中，CQL [Kumar et al., 2020] 算法显著好。而在 Offline Model Selection，作者得出的结论是，即使挑选了在 Online Model Selection 中最好的一部分模型，它们看起来也和随机的性能相当
  * [想法] 落地部分有太多的脏活要干，人们对它们知之甚少，因此现在困在铁屋子里。现实世界依然只有推荐系统，AI 这类 overfit 的场景才好上 RL
### 2022 - [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://openreview.net/pdf?id=AXDNM76T1nc)
  * 这篇论文研究了如何用大量的 minecraft 未标注游戏视频来帮助强化学习，实现一些非常难的任务（比如挖钻石）
  * 这篇 paper 有 9 个作者，说是至少都全职投入了半年，完整训练一次需要 “9 days on 720 V100 GPUs”，附录二十多页
  * 收集数据：随便从网上找 270k hours 数据，然后过滤一下，去掉那些水印，主播露脸，创造模型，非 pc 平台的数据，得到 70k hours 的干净数据；训练一个逆动力学模型 IDM：找一些人来玩 2k hours，记录键盘鼠标操作，然后用这些标注蒸馏出一个从 video 反推键鼠操作的模型；用 IDM 给所有视频打标签，开始模仿学习；经过模型学习预训练后，模型已经表现出相当的智能（会做很多基本的事情，比如砍树、游泳、打猎、跳柱）；然后作者研究这样的预训练对于下游任务的作用
  * 注意这整篇 paper 定义的动作空间完全是键鼠和人机界面，和以往简化动作空间的做法不同，极大提高了任务难度
  * 直觉上学 IDM 确实是个简单一些的任务，完全的监督学习
  * IDM 是给前后帧估计键鼠操作，BC 是给前帧预测未来所以更困难；之后的消融实验表明，虽然听起来简单，但 IDM 至少也需要 100 hours 的数据来训练
  * 文中还用到了 “earlygame_keyword” 和 “contractor_house” 子数据集，前者是在收集数据的时候选用了特定的关键字，后者是找人来玩的时候，让他们十分钟造一个简单房子，在这俩数据集上 finetune，会使得模型产生更多的特定行为
  * KL loss 可以约束训练后的模型和最初的预训练模型的 gap，这样使得模型避免 “灾难性遗忘”（比如模型可能看过怎么使用一个熔炉的界面，但是因为在 finetune RL 阶段要很后期才会出现熔炉，就忘了怎么用）
  * 这个模型制造工具的水平相当高，10分钟有2.5%都造出了钻石镐，感觉超越大部分玩家
  * 减少数据量会使得一些行为无法学习
  * [想法] 结果令人震撼，各种投入都拉满了，在 RL 领域少见的暴力，力大砖飞（用很朴素的方法实现了很强的结果）
### 2021 - [What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study](https://openreview.net/pdf?id=nIAxjsniDzg)
  * 本文是 PPO 时代的又一实践指导工作，给出的一些建议相当有用，本文宣称至少考虑了实现中的 50 种选择和包含 250’000 组实验验证
  * 附录有 50 页可以用来当调参手册
  * 不管是从鲁棒性还是最优性考量，PPO 是 actor-critic 中最好的
  * 总体来说 actor-critic 不共享参数比较好，policy 网络的初始值有出乎意料的巨大影响
  * 对于规范化技术，优先考虑 observation normalization，其次考虑 gradient clipping
  * 调大 num_epochs，打乱多个环境收集的样本，并且 per data pass 计算 adv
  * gamma 需要精细调整，用 Adam with β1 = 0.9，精心选一个学习率，通常 3e-4 比较安全
  * [想法] 可惜这篇不做 atari，在我心目中 atari 的场景多样性好很多
  * [想法] RL 和 GAN 在 TRPO 及其前时代恶名远播，这几年社区不仅有更稳定的算法（PPOv2），还有更多开源的优秀实现和很多经验总结，使得 RL 至少在实验场景下变得更加可用了
### 2019 - [Learning to Paint with Model-based Deep Reinforcement Learning](https://openaccess.thecvf.com/content_ICCV_2019/html/Huang_Learning_to_Paint_With_Model-Based_Deep_Reinforcement_Learning_ICCV_2019_paper.html)
  * 一个试图联系视觉和强化学习的工作
  * 使用深度强化学习模型来解决绘画任务，模型能将一张图像解耦为成百的笔画序列，并生成一个与目标图像相似的绘画作品（抽象和临摹）
  * 使用了神经网络绘制器来实现高效的绘画过程，提高了模型的性能
### 2018 - [Playing text-adventure games with graph-based deep reinforcement learning](https://arxiv.org/abs/1812.01628)
  * 在强化学习的探索过程中学习 knowledge graph，可以用来对动作空间进行剪枝
  * OpenIE 会自动构建出 ⟨subject, relation, object⟩ 的三元组关系，感觉是对自然语言的一种显式的逻辑化解读
  * 作者把找出的所有三元组按一个预设规则生成图，对所有 action 进行得分计算，可以看作 action 和图关系的一个关联度计算
  * [想法] 我觉得构造一种 action space 剪枝的方法会很有趣，但是可能按 nn 的思路会做的更隐式一点，比如说鼓励 agent 尝试一些让环境变化比较大的动作
## Image Generation
### 2022 - [One-Shot Adaptation of GAN in Just One CLIP](https://arxiv.org/pdf/2203.09301.pdf)
  * 以往将一组图片变成 novel domain 的工作，仅靠一张 novel domain 的图片不太 work（难点在于只有一张 target 的时候，GAN 训不好）；作者基于 CLIP 设计了一些 loss function，取得了不错的效果
  * 对于一个 pretrained generator G，在它的编码空间内找一个 w，使得 G(w) 和 target 在各种语义下接近，相当于找到了一个 target 的编码
  * 第二步作者在开集上 fine-tune 这个 G，用一致性约束；即对于两个 w，它们经过 G 和 finetuned G 的解码后，在 CLIP 语义空间上的距离相似
  * [想法] 感觉就作者做的这件事情来说，Splice 更强；特别是本文依赖于 pretrained GAN，大多数效果由 VQGAN + CLIP 或者 StyleCLIP 也能出的差不多的感觉
### 2022 - [Splicing ViT Features for Semantic Appearance Transfer](https://arxiv.org/abs/2201.00424)
  * 训练一个 UNet 来做纹理迁移
  * Dino 对图片推理得到的 cls token 经过反演，会得到一个结构破坏但内容保留的图片
  * 图 A，图 B，G(B)，让 A 和 G(B) 外观像，B 和 G(B) 结构像（设计了两种基于预训练的 dino 的 loss，简单理解就是初版 styletransfer）
  * [想法] G 可以很容易地适应不同的B，但是目前因为 A 不是 G 的一个输入，还不能 general 地生成 G(A, B)
## Others
### 2021 - [D^2LV: A Data-Driven and Local-Verification Approach for Image Copy Detection](https://arxiv.org/pdf/2111.07090.pdf)
  * 比赛报告，这个比赛给了一个 query 集（5w张）和 ref 集（100w张），问 query 集的每一张图片，是否抄袭了 ref 集 (Copy detection)，抄袭定义为用 ref 集的某张图做素材，ps 进自己的图里，给出置信度
  * 这是一个变种的图像检索任务，第一名的方案除了比较标准的预训练 + 对比学习以外，用了大量的图像增广和非常疯狂的 ensemble 策略
  * 作者说虽然用无监督训练的方法直接解 copy detection 是自然的，但是由于两个原因放弃了 1）原始BYOL 8卡训了两周，再加点增广可能就做不起了 2）训不出来。所以作者用了原始的 BYOL 和 Barlow Twins 预训练模型
  * 几十种数据增广，真的很疯狂
  * 最终作者用了 33 个模型集成（包括不同大小的 resnet 和 IBN 变种），在 3 个图像尺度下多对一查询，图片一对多查询实在做不起了，只跑了三个模型
  * [想法] 我起初完全没想到这个比赛能把 AP 刷到 90+，还是低估了暴力的威力
### 2021 - [Simple but Effective: CLIP Embeddings for Embodied AI](https://arxiv.org/pdf/2111.09888.pdf)
  * CLIP 的 ResNet50 放在 Embodied AI 的一众下游任务上，可以达到降维打击的效果。Embodied AI 主要是关注机器人和环境的交互
  * 分析 CLIP 相对 imagenet 训练模型的优势，通过研究四个基础任务 “object presence, object presence at a location, object reachability, and free space”
“free space” 字面不好理解，定义为 “predict how many steps an agent can move forward given an egocentric view”
  * 看起来 CLIP 主要提升了 “Object Localization” 的性能
  * 作者认为不能靠 imagenet 分类精度来评判模型在下游任务表现
  * [想法] 全文贯彻了 Simple but Effective 的理念，而且关注在 Embodied AI 领域，但是我们仍然有很多问题想知道，比如 finetune CLIP 有没有好处，CLIP 还能让什么任务受益？
### 2019 - [Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization](https://arxiv.org/pdf/1804.06516.pdf)
  * 对于车辆更有道理的数据增广，借助渲染器改改光照颜色纹理等
  * 给车贴上随机的纹理，放到随机光照的随机场景中。再在场景中加入一些奇怪的漂浮物
  * [想法] 个人觉得 DR 和后来的 adversarial training，texture dropout 殊途同归，强迫 nn 关注某一些不变的特征
