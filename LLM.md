## LLM
### 2024 - [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5)
  * 背景：像人类一样证明几何问题的搜索树，在“不加辅助线”的情形下是有限甚至小计算资源下可穷尽的，但加辅助线的方法是几乎无穷无尽的，这是几何定理证明器长期来的主要难点。
  * 本文构建了一个符号推断 + LLM 的证明系统。这套系统的测试集是从 IMO2000 以来的 30 个几何证明题（总几何题的75%，原因是其它题很难形式化表示），GPT4 做出的题目数是 0。AlphaGeometry (AG) 做出了 25 题，并且对其中某道题的前提进行了减少。做不出来的主要原因是，人类会采用更 high-level 的证明手段（我理解是用一些现成定理）：在相对简单的问题上，人类平均得分和AI生成的证明长度之间没有显著相关性。AG 给出的证明可以很容易地转成自然语言，得到了 IMO 评审组的认可。
  * 模型自动构建的数据集中，有 9% 带有辅助线，最长的步骤达到 250 步。
  * 让大语言模型发掘“加辅助线”的规律。关键是合成数据的技巧。对于一个随机的几何图形（作者写了一个生成器），穷举它能够得到的命题，然后回溯每个命题的表述它要用的几何对象集合，和表述【证明它要用的最小前提集】要用到的几何对象集合。两个集合的差就是辅助线。
  * 总数据是1亿个几何证明（从原始5亿数据中筛选去重，5亿数据用10w个cpu跑了三天），其中有900万个带辅助线的题目。作者检查了数据泄露：IMO 的题和生成数据没有交集，jgex_ag_231 中约有 10% 和训练集重合。
  * 1.51亿参数 (0.15B) 模型，在<命题，结论，证明>上进行了预训练和微调。
  * 一些具体技术问题：1. 对图表进行采样的操作空间是什么？ 我们需要一个动作空间，让我们能够轻松地采样有趣的图表（Extended Data Table 1 ）；2. 推导出所有事实的符号算法是什么？ 建立在之前的工作上(structured DD)，并大大增强了现有算法能力：使用 algebraic reasoning 来推导距离、角度和比例。作者实现的 DD + AR 过程可以在几秒到几分钟之内完成，AR 的实现主要是高斯消元；3. 将图简化为最小形式的算法是什么？ 超级不平凡 - 但本文解决了它！（针对 DD 和 AR 都有很多算法设计，比如解最小覆盖）。还有一些额外的细节，比如在找到证明路径以后，简化并且转成自然语言。
  * 具体训练：在全数据集上进行预训练(21->25)，并在带辅助线的数据上 finetuning (23->25)，预训练总计采 1.6 亿数据量，微调采 1600 万。这里作者强调，让 LLM 在证明过程上预训练，可以增强它构造辅助线的能力。
  * 作者的 DD+AR 可以解决 14 题，加上 LLM 生成辅助线解决另外 11 题。推理时使用 beam search，即使只采用 beam size = 8 (2% 计算量，versus 512），也能解决 21 个题。
  * 经过symbolic system生成verified data会显著提高LLM性能
  * [想法] 几何恰好有一个很好的动作空间来采样有趣的问题。 但对于一般数学领域，随机抽样可能不是正确的方法。 弄清楚如何从现有的人类数据中引导并了解有趣问题的分布是更有前途的。
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