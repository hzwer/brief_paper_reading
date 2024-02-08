# briefly_paper_reading
## LLM
* [xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval](https://arxiv.org/pdf/2303.03004.pdf)
  * 收集了 codeforces 的 7.5k 题目和相关的 5M+ 解答，构造七个任务（检索，翻译等），分析 chatGPT 的表现
  * 数据集的特点是量大、题多、语言多、有部分测例
  * codeforces 比 humaneval 难很多
  * chatGPT 在 codeforces 1600 分段有 10% 的通过率，而我的常识中 chatGPT 很难做对 1200 分及以上的题。作者发现 chatGPT 对于某个时间点后的题正确率陡降，说明应该是背过题库

* [Aligning AI With Shared Human Values](https://arxiv.org/pdf/2008.02275.pdf)
  * 构建了一个伦理相关数据集，commonsense, deontology, justice, utilitarianism, and virtue. 即常识、义务论、正义、功利主义和美德。在这些定义下，可以构建争议较小的道德场景判断
  * 题外话：MMLU 评测集中的道德情景，GPT4 表现相当好
  * 作者考虑的道德场景是一般情形下的，举例：闯入另一个人的住所可能是不对的，但是消防员在特殊情形下可以这样做
  * 作者采用众包的形式来对数据进行标注，确保多个人给出一致意见，否则数据会被丢弃。标注人员主要来自英美，作者也评估了用10个印度标注员小规模重标，一致性达到 93% 以上，其中错误的部分还可能来自对习语的误解
  * 正义包含两个部分，一个是 Impartiality（公正），一个是 Desert（应得的）：前者是说一个人所受到的对待不应该由于偏见和歧视被改变，后者是说一个人得到的对待和他的行为是相对对等的
  * [想法] 我在为每个子集编写测试 prompt 的过程中，用 gpt4 来测试可以得到很好的反馈，帮助把 prompt 写的更清晰和消歧义
  * [想法] 大模型自然表现出比小模型更高的道德水平，也有一些文献表明大模型具有自主降低输出毒性的能力
