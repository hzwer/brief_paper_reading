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