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
