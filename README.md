# briefly_paper_reading
## LLM
* 2024 - [xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval](https://arxiv.org/pdf/2303.03004.pdf)
  * 收集了 codeforces 的 7.5k 题目和相关的 5M+ 解答，构造七个任务（检索，翻译等），分析 chatGPT 的表现
  * 数据集的特点是量大、题多、语言多、有部分测例
  * codeforces 比 humaneval 难很多
  * chatGPT 在 codeforces 1600 分段有 10% 的通过率，而我的常识中 chatGPT 很难做对 1200 分及以上的题。作者发现 chatGPT 对于某个时间点后的题正确率陡降，说明应该是背过题库
* 2021 - [Aligning AI With Shared Human Values](https://arxiv.org/pdf/2008.02275.pdf)
  * 构建了一个伦理相关数据集，commonsense, deontology, justice, utilitarianism, and virtue. 即常识、义务论、正义、功利主义和美德。在这些定义下，可以构建争议较小的道德场景判断
  * 题外话：MMLU 评测集中的道德情景，GPT4 表现相当好
  * 作者考虑的道德场景是一般情形下的，举例：闯入另一个人的住所可能是不对的，但是消防员在特殊情形下可以这样做
  * 作者采用众包的形式来对数据进行标注，确保多个人给出一致意见，否则数据会被丢弃。标注人员主要来自英美，作者也评估了用10个印度标注员小规模重标，一致性达到 93% 以上，其中错误的部分还可能来自对习语的误解
  * 正义包含两个部分，一个是 Impartiality（公正），一个是 Desert（应得的）：前者是说一个人所受到的对待不应该由于偏见和歧视被改变，后者是说一个人得到的对待和他的行为是相对对等的
  * [想法] 我在为每个子集编写测试 prompt 的过程中，用 gpt4 来测试可以得到很好的反馈，帮助把 prompt 写的更清晰和消歧义
  * [想法] 大模型自然表现出比小模型更高的道德水平，也有一些文献表明大模型具有自主降低输出毒性的能力
## Video
* 2020 - [Softmax Splatting for Video Frame Interpolation](https://arxiv.org/abs/2003.05534)
  * 主要是提出了 softmax splatting，实现一种 forward-warp
  * 在 forward-warp 时，处理多个像素映射到一个像素点的方法，直接相加会溢出，可以让网络预测一种加权平均，文中对比发现 softmax 比较好
  * 很大部分涨点其实来自于warp context（用光流去 warp pyramid feature map），把它们加入到最后的 U-net 里
  * 有大量 trick，比如 laplacian loss，U-net 的改版 Grid-net 等
* 2020 - [MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://arxiv.org/pdf/2003.10955.pdf)
  * pwc-net 中，使用两个 warped feature 计算 cost volume
  * 从消融实验看，主要是这个 Asym-conv 涨的点，可能说明 warped feature 与一般 feature 直接进行匹配是不合适的 （作者称：Intuitively, the warping operation induces
ambiguity to the occluded areas and breaks the symmetricity of the feature matching process, so an asymmetric design might be helpful to conquer this divergence.）
* 2020 - [ScopeFlow: Dynamic Scene Scoping for Optical Flow](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bar-Haim_ScopeFlow_Dynamic_Scene_Scoping_for_Optical_Flow_CVPR_2020_paper.pdf)
  * 从训练集中裁出小图训练，中间的像素被选出的概率大，而边上的像素被选出的概率小，这种偏差对图片光流训练会有很大影响
  * 统计发现，在几个光流训练集中，运动比较小的物体通常在图片中间（远景），而运动大的物体集中在图片边缘。
* 2016 - [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/pdf/1612.01925.pdf)
  * 使 CNN 光流方法达到了传统光流法的效果
  * 喂不同难度数据的顺序影响模型训练效果
  * 先出一次光流，用光流把 img2 向 img1 warp，然后再把两张图放到下一个网络，级联一些小网络，逐步出残差 refine 之前的光流可以涨点，现在这都是标准操作了
  * 对于小范围的位移构造了一个数据集 ChairsSDHom，并针对改进新的网络结构
  * FlyingChairs, 22k 张图，背景是512*384的flickr图，前景是809种62视角的3d椅子，椅子只有2d移动
  * FlyingThings3D，22k 张图，考虑光线，3d移动，各种物体
  * 直接在FlyingThings3D训不好，先在FlyingChairs上训，再在FlyingThings3D上finetune比较好
  * 尝试用得到的光流来辅助 motion segmentation 和 action recognition 的模型
