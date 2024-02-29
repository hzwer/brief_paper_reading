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
