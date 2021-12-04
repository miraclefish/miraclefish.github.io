# Deep Semi-supervised Learning [A Survey]


## Reference

> 1. [Yang X, Song Z, King I, et al. A Survey on Deep Semi-supervised Learning[J]. arXiv preprint arXiv:2103.00550, 2021.](https://arxiv.org/pdf/2103.00550)

## 概述

本篇综述不涉及关于SSL综述[代表作](https://ieeexplore.ieee.org/abstract/document/4787647)的内容，而是主要针对基于DL的算法。

### SSL任务的分类

- Semi-supervised classification
- Semi-supervised clustering
- Semi-supervised regression

### DSSL模型的分类

- Generative methods
- Consistency regularization methods
- Graph-based methods
- Pseudo-labeling methods
- hybrid methods

![DSSL分类](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/taxonomy.png 'DSSL分类')

## 背景介绍

数据集表示为 $X=\{X_L, X_U\}$ ，其中 $X_L=\\{x_i\\}^L_{i=1}$ 是一个比较小的有标注的子集，标注为 $Y_L=(y_1, y_2,\dots,y_L)$ ，而 $X_U=\\{x_i\\}^U_{i=1}$ 是一个比较大的无标注的子集，通常假设 $L \ll U$ 。

假设数据集总共包含 $K$ 个类别，则 $X_L$ 被标注为 $\\{y_i\\}^L_{i=1}\in(y^1,y^2,\dots,y^K)$ ，则SSL需要去求解如下的优化问题：
$$
\min _{\theta} \underbrace{\sum_{x \in X_{L}, y \in Y_{L}} \mathcal{L}_{s}(x, y, \theta)}_{\text {supervised loss }}+\alpha \underbrace{\sum_{x \in X_{U}} \mathcal{L}_{u}(x, \theta)}_{\text {unsupervised loss }}+\beta \underbrace{\sum_{x \in X} \mathcal{R}(x, \theta)}_{\text {regularization }}
$$

- $\mathcal{L}_s$ 表示每个样本的监督损失
- $\mathcal{L}_u$ 表示每个样本的无监督损失
- $\mathcal{R}$ 表示每个样本的正则化项（一致性损失或者是设计出的`regularization term`）

> 注意：$\mathcal{L}_u$ 和 $\mathcal{R}$ 通常没有严格区分，因为 $\mathcal{R}$ 一般也是与标注信息无关的

根据测试集数据是否提供将SSL分为两种setting：

- Transductive learning (Graph-based methods)
- Inductive learning (else methods)

### SSL的相关假设

1. **Self-training assumption**：Self-training model的预测通常是正确的，因为该假设成立的话，high-confidence predictions就可以被当作真实标签。这个假设通常在类分离的比较好的时候满足。
2. **Co-training assumption**：不同合理的假设导致不同的有标签数据和无标签数据的组合。Blum提出一种co-training model，模型满足“实例 $x$ 有两个条件独立的 `views` ，每个 `view` 满足一个分类任务”。
3. **Generative model assumption**：已知先验 $p(y)$ 和条件分布 $p(x|y)$ 的情况下，可以通过 $p(x,y)=p(y)p(x|y)$ 将无标签数据和类别有效连接起来。
4. **Clustering assumption**：同类样本在高维空间中通常成簇聚集，同类的两个样本之间用短线连接通常不会穿过低密度区域，因此大量的无标签数据可以用来调整分类边界。
5. **Low-density separation**：与Clustering assumption类似，决策边界不应穿过高密度区域。
6. **Manifold assumption**：如果 $x_1$ 和 $x_2$ 在低维流形上局部相邻，他们将拥有相同的类别。这反映的是决策边界的局部平滑性。

### Classical methods

1. 1970年，SSL概念首先被提出，实现思路是采用 self-learning 的方式，先拿有标签的数据训练模型，再拿无标签的数据测试，prediction 值较高的认为是该样本的 ground truth label (pseudo label)，把这部分再拿进模型训练，逐步迭代。
2. 生成模型依据假设 $p(x,y)=p(y)p(x|y)$ ，对于有标签数据， $p(y)$ 和 $p(x,y)$ 已知，需要求解 $p(x|y)$ 的参数，$p(x|y)$ 通常为可识别的分布（多项式分布、高斯分布）。通常用 EM 算法迭代求解。 但可识别的分布通常不能完全匹配真实分布，会导致分类性能下降。
3. TSVMs (Transductive Support Vector Machines)：和SVMs同样，TSVMs优化数据点和决策边界之间的gap，然后借助无标签数据的信息expand the gap。
4. Graph-based methods 依赖于有标签数据和无标签数据分布的几何结构（流形）。通过探索数据的图或流形结构，可以借助非常少的标签 propagate information 来学习。（如：Label propagation）

### 相关学习范式

- 迁移学习：Transfer Learning
- 弱监督学习：Weakly-supervised Learning
- Positive and unlabeled learning
- 元学习：Meta Learning
- 自监督学习：Self-supervised Learning

### 数据集和应用



![相关数据集和应用](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/Datasets.png "相关数据集和应用")



## Generative Methods

### Semi-supervised GANs

![GAN模型的演化关系](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211116153818.png "GAN模型的演化关系")

众所周知，GAN的损失函数表示如下：
$$
\min _{G} \max _{D} V(D, G) = \mathbb{E} _{x \sim p(x)} [\log D(x)] + \mathbb{E} _{z \sim p _{z}} [\log (1-D(G(z)))]
$$
![GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211116154949.png "GAN")

因为GAN可以从无标签数据中学习数据的真实分布，所以它可以用于SSL任务，而具体的应用方式分为如下四种：

- 重用（re-use） Discriminator 的特征
- 使用 GAN 生成的样本去正则化（regularize）分类器
- 学习一个 inference model
- 使用 GAN 生成的样本作为另外的训练数据

#### CatGAN

**Categorical Generative Adversarial Network**

![CatGAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211116155042.png "CatGAN")

把 GAN 的 Discriminator 改成了分类器（而不是仅判断生成的样本是否真实）。个人理解主要目的有两个：

- 可以实现无监督的聚类，因为生成的样本会被判别器主动分类，只不过类别个数需要手动设定；
- 可以实现半监督学习，利用有标签数据同时训练模型，因为 Discriminator 可以输出分类。

{{<admonition type=note title="Loss components">}}

*  $H[p(y \mid x, D)]$ ：对于无标签数据，最小化条件概率的熵，即 $D$ 需要对 $x$ 给出更明确地类别输出；
*  $H[p(y \mid G(z), D)]$ ：对于生成的数据，最大化条件概率的熵，即 $D$ 需要对 $G(z)$ 给出近似均匀分布的输出（GAN的思想）；
*  $H[p(y\mid D)]$ ：最大化类别分布先验的熵，是在各类样本个数基本相等的假设下，保证 $D$ 对样本的聚类没有偏见。这里是对数据集中所有样本去计算的，所以无法与前面的两个损失在一个 batch 里面兼容，文中对它的计算做了特殊处理。
*  $C E[\mathbf{y}, p(y \mid \mathbf{x}, D)]$ ：有标签数据的交叉熵，用于半监督学习的设定。

> 注意：这里提到的损失函数最大或者最小化要根据模型训练在 $G$ step 还是在 $D$ step 来确定，满足GAN的训练规则。

{{</admonition>}}

#### CCGAN

**Context-Conditional Generative Adversarial Network**

![CCGAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211116172350.png "CCGAN")

主要亮点是利用图像周围的像素（context information）学习图像特征。图中的 $m$ 是一个二值的 mask，用于 drop out 图像中的 specific portion （比如一个方形区域）。$x_{I}=(1-m) \odot x_{G}+m \odot x$ 表示 in-painted image，就是根据输入的挖空图像生成的补全图像。其中 $x_{G}=G(m \odot x, z)$ 表示根据挖空图像和噪声生成的图像。其余部分与 GAN 相同，个人觉得类似 Transformer 中的掩码机制。

#### Improved GAN

**Improved Techniques for Training GANs**

![Improved GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211117103735.png "Improved GAN")

与 CatGAN 相比，ImprovedGAN 在判别器 $D$ 的输出上多加了一类，即第 $K+1$ 类，表示 $G$ 生成的数据。直观理解就是，$D$ 本来是要判别样本是 real sample 还是 generated sample 的，但是 CatGAN 把它改成了分类器，即默认 generated sample 一定属于 $K$ 类。这显然是有问题的，如果生成的数据压根啥都不是呢，所以给 $D$ 多加一类，强制 $G$ 要生成与真实样本相似的前 $K$ 类的数据。此外，这篇文章还提出了很多训练 GAN 的技巧，这里做一下列举，详细的请参考[原文](https://arxiv.org/abs/1606.03498)。

{{<admonition type=note title="Techniques">}}
* Feature matching
* Minibatch discrimination
* Historical averaging (这个很像自监督学习里的MoCo方法)
* One-sided label smoothing
* Virtual batch normalization
{{</admonition>}}

#### GoodBadGAN

**Good semi-supervised learning that requires a bad GAN**

![GoodBadGAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211117103735.png "GoodBadGAN")

这篇文章意识到 Improved GAN 的 $G$ 和 $D$ 可能不能同时达到最优（即判别器达到了最优效果，但是生成器可能产生不真实的样本）。该方法从理论上证明了，为什么 bad samples 可以增强 SSL 的 performance。这里给出一个直观解释，理论的证明参考[原文](https://arxiv.org/pdf/1705.09783.pdf)。

> 对于一个 Perfect Generator，训练的目标是期望它与真实数据的分布一致：$p_{G}=p$ 。但是如此一来会导致 $D$ 的最优解等价于有监督损失的最优解（原文中有理论证明），即无标签的数据失效了。而一个 Complete Generator，是应该能够产生 Bad samples 的，因为这样可以使 $G$ 生成的部分 Bad sample 填充高维空间中的低密度区域，使得 $D$ 的 boundary 不会落在类内的低密度区域，避免分界面穿过流形。

{{<admonition type=note title="Loss components">}}

* $\mathbb{E} _{x \sim p _G} \log p(x) \mathbb{I}[p(x)>\epsilon]$ ：$G$ 的优化目标（最小化），用于生成 bad samples 的惩罚项，它只对高密度区域的样本起作用，低密度区域的样本不受影响（$\mathbb{I}[\cdot]$ 是示性函数）。这里是指对于 $p(x)>\epsilon$ 的样本，约束使其 $p(x)$ 越小越好，直至 $p(x)<\epsilon$ 后即可消除惩罚（该样本的惩罚等于0）
* $\mathbb{E} _{x \sim \mathcal{U}} \sum _{k=1}^{K} p _{D}(k \mid x) \log p _{D}(k \mid x)$ ：$D$ 的优化目标（最大化：这里是负的条件熵，相当于最小化条件熵），用于使无标签数据的输出尽可能地确定，避免 $D$ 对于无标签数据的分类结果过于均匀。

{{</admonition>}}

#### Localized GAN

**Global versus Localized Generative Adversarial Nets**

![Localized GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211117173942.png "Localized GAN")

通常的 GAN 是指用生成器 $G$ 从随机噪声 $z\in\mathbb{R}^N$ 生成样本 $G(z)\in \mathbb{R}^D$ ，在这种情况下，生成样本 $G(z)$ 的环境空间（ambient space）是用 $N$ 维的全局坐标系 $z$ 来表示的。所有生成的样本会形成一个 $N$ 维的 manifold $\mathcal{M} = \\{ G(\mathbf{z}) \mid \mathbf{z} \in \mathbb{R}^{N} \\}$ 。这样的假设有两个缺陷：

{{<admonition type=warning title="问题">}}
* 在全局坐标系的假设下，样本点 $x$ 的局部结构无法直接得到，因为流形空间是 $N$ 维的，$D$ 维的样本空间只是一个 Embedding，所以必须知道 $G^{-1}(\cdot)$ 才能从 $x$ 映射回 $z$ 从而知道 $x$ 的局部结构。
* 如果 $x$ 的维数有缺陷（$\mathcal{T} _x < N $），则切空间 $\mathcal{T} _{x}$ 会产生局部塌陷（local collapse）。如此一来，当 $z$ 在某些方向上发生改变时，$G(z)$ 就不会再产生有意义的数据点（即 $x$ 不再变化）。
{{</admonition>}}

所以，本文提出 local generator $G(x,z)$ 满足如下两个条件：

{{<admonition type=note title="Conditions">}}
* _locality_ ：$G(x, 0)=x$ , i.e., 局部坐标 $z$ 的原点必须在 $x$
* _orthonormality_ ：$\mathbf{J} _{x} ^{T} \mathbf{J} _{x}=\mathbf{I} _{N}$ , i.e., $\mathcal{T}_x$ 的基必须是标准正交的，保证 $x$ 的局部不会产生塌陷。
{{</admonition>}}

通过最小化如下的正则项来约束这两个条件：
$$
\Omega_{G}(\mathbf{x})=\mu\|G(\mathbf{x}, \mathbf{0})-\mathbf{x}\|^{2}+\eta\left\|\mathbf{J}_{\mathbf{x}}^{T} \mathbf{J}_{\mathbf{x}}-\mathbf{I}_{N}\right\|^{2}
$$
本文通过解决流形上函数的求导问题（详情见[论文](https://arxiv.org/pdf/1711.06020.pdf)），在半监督学习任务上得到了一个局部一致的分类器。

#### CT-GAN

![CT-GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211123115129.png "CT-GAN")

结合 Consistency training 和 WGAN 用于半监督学习，依赖于 Lipschitz 连续性条件。一致性约束通过对样本添加两次扰动来实现：（1）输入位置加扰动；（2）hidden layer 加扰动。细节参考[论文](https://arxiv.org/pdf/1803.01541.pdf)。

#### BiGAN

**Bidirectional Generative Adversarial Networks**

![BiGAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211123120158.png "BiGAN")

双向GAN，顾名思义，即是两个方向的生成，从 $z$ 到 $x$ 是 $G(z)$ 表示 generator，从 $x$ 到 $z$ 是 $E(x)$ 表示 encoder。$D$ 对 $(G(z), z)$ 和 $(E(x), x)$ 同时进行判别，迫使 $G$ 和 $E$ 能够学习出一对可逆的映射。目标函数如下：
$$
\min _{G, E} \max _{D}  \mathbb{E} _{\mathbf{x} \sim p _{\mathbf{X}}}\left[\mathbb{E} _{\mathbf{z} \sim p _{E}(\cdot \mid \mathbf{x})}[\log D(\mathbf{x}, \mathbf{z})]\right]+\mathbb{E} _{\mathbf{z} \sim p _{\mathbf{Z}}}\left[\mathbb{E} _{\mathbf{x} \sim p _{G}(\cdot \mid \mathbf{z})}[\log (1-D(\mathbf{x}, \mathbf{z}))]\right]
$$
此文偏理论，对其进行了详细的[证明](https://arxiv.org/pdf/1605.09782v7.pdf)。

#### ALI

**[Adversarially Learned Inference](https://arxiv.org/abs/1606.00704)**

ALI和BiGAN的结构基本一样，只是把 Encode 表述成了 Inference。

#### Augmented BiGAN

相比 BiGAN，做出了两点改变：

* 使用了 Feature matching loss：$\| \mathbb{E} _{x \in X} D(E(x), x)- \mathbb{E} _{z \sim p(z)} D(z, G(z)) \| _{2}^{2}$ 
* 防止生成的样本（$G(E(x))$）类别发生改变，引入损失项：$\mathbb{E} _{x \sim p(x)}[\log (1-D(E(x), G_x(E(x))))]$

#### Triple GAN

![Triple GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211123220823.png "Triple GAN")

将传统的GAN改成了一个 [three-player game](https://arxiv.org/pdf/1703.02291.pdf)：

{{<admonition type=note title="Three players">}}

* $G$ ：generator 用一个 conditional network 去生成对应真实标签的假样本；
* $C$ ：classifier 给样本生成伪标签；
* $D$ ：discriminator 区分一个 data-label 对是否来自真实的数据集。

{{</admonition>}}

损失函数：
$$
\begin{aligned} \min _{C, G} \max _{D} U(C, G, D)= & E _{(x, y) \sim p(x, y)}[\log D(x, y)]+\alpha E _{(x, y) \sim p _{c}(x, y)}[\log (1-D(x, y))] \\\\ & +(1-\alpha) E _{(x, y) \sim p _{g}(x, y)}[\log (1-D(G(y, z), y))] \end{aligned}
$$

- 第一项对应真实样本和真实标签，$D$ 需要判别其为真；
- 第二项对应真实样本和其伪标签，$D$ 需要判别其为假，迫使 $C$ 能够给出真实的标签；
- 第三项对应生成样本和生成标签，$D$ 需要判别其为假，迫使 $G$ 能够根据标签生成真实的样本。

#### Enhanced TGAN

[Enhanced TGAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Enhancing_TripleGAN_for_Semi-Supervised_Conditional_Instance_Synthesis_and_Classification_CVPR_2019_paper.pdf) 在 Triple GAN上做了修改：

- Generator 生成图像时根据类别分布生成（conditioned on class distribution），并且加上了逐类的平均特征匹配（class-wise mean feature matching）
- 分类网络包括两个分类器协作学习，为 $G$ 的训练提供更多的类别信息
- Semantic matching term：增强语义一致性

#### Margin GAN

[Margin GAN](https://openreview.net/pdf?id=HJl1gSBeUS) 是 Triple GAN 的另一个扩展模型：

![Margin GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211124163824.png "Margin GAN")

与 Triple GAN 相似，有三个部分（$G$ , $D$ , $C$）的对抗训练，但是不同的是，它引入了另一种对抗训练方式：

> Generator 企图生成 large-margin 的假样本；Classifier 企图对这些假样本产生 small-margin 的分类结果。

此外，Wrong pseudo label 通常会对基于伪标签的方法产生很坏的影响，而 Margin GAN 提升了模型对 wrong pseudo label 的容忍度，并且由于 $C$ 的约束， $G$ 生成的假样本更容易落在”正确“的决策边界附近，细化和缩小了真实样本周围的决策边界。如下图产生的问题，当 $G$ 能够在接近真实的决策边界处生成 fake samples 时，wrong pseudo label 就更不容易落在 real samples 中，和 GoodBadGAN 的思路类似，即一个坏的 $G$ 会对分类器的结果产生好的影响。

![Wrong pseudo labels problem](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211125151029.png "Wrong pseudo labels problem")

{{<admonition type=note title="什么是 Margin ?">}}

* 对于 Unlabeled 样本来说，其 Margin 表示为 $|f(x)|$ ，它表示分类器认为目前的预测是正确的并且让分类器更加确信当前的预测，它可以降低泛化误差的上界，带来更好的泛化性能。（这里没太懂，可以参考[论文](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.3613&rep=rep1&type=pdf)）
* 对于有标签的样本来说，其 Margin 表示为 $\operatorname{Margin}(x, y)=C _{y}(x)-\max _{i \neq y} C _{i}(x)$ 即分类器对真实标签生成的概率减去对错误标签生成的概率的最大值。可以直观理解为分类器的分类性能，越准确，Margin 越大。

{{</admonition>}}

**Discriminator：**
$$
\operatorname{Loss}(D) = -\\{E _{x \sim p^{[l]}(x)}[\log (D(x))]+E _{\widetilde{x} \sim p^{[u]}(\widetilde{x})}[\log (D(\widetilde{x}))]+E _{z \sim p(z)}[\log (1-D(G(z)))]\\}
$$
第一项是有标签数据，第二项是无标签数据，第三项是生成数据。

{{<admonition type=note title="Classifier">}}

1. 有标签数据：$\operatorname{Loss}\left(C^{[l]}\right) =E_{(x, y) \sim p^{[l]}(x, y)}\left[-\sum_{i=1}^{k} y_{i} \log \left(C(x)_{i}\right)\right]$ 交叉熵损失，对应 large margin 的目标；
2. 无标签数据：$\operatorname{Loss}\left(C^{[u]}\right) =E_{\widetilde{x} \sim p^{[u]}(\widetilde{x})}\left[-\sum_{i=1}^{k} \widetilde{y}_{i}^{[u]} \log \left(C(\widetilde{x})_{i}\right)\right]$ 交叉熵损失，只不过这里的 label 是伪标签经过 One-hot 编码之后得到的，同样对应最大化 Margin；
3. 生成数据：$\operatorname{Loss}\left(C^{[g]}\right)=E_{z \sim p(z)}\left[\operatorname{Loss}_{I C E}\left(\widetilde{y}^{[g]}, C(G(z))\right)\right]$ 这里的 ICE 损失是逆交叉熵（Invert cross entropy），即 $\operatorname{Loss}_{I C E}\left(\widetilde{y}^{[g]}, C(G(z))\right) = -\sum_{i=1}^{k} \widetilde{y}_{i}^{[g]} \log \left(1-C(G(z))_{i}\right)$。 目的是为了减小 Margin，使得 $G$ 生成的数据更趋于均匀分布，即填充数据中的低密度区域，使得决策边界落在这些低密度区域。
4. 最终：$\operatorname{Loss}(C)=\operatorname{Loss}\left(C^{[l]}\right)+\operatorname{Loss}\left(C^{[u]}\right)+\operatorname{Loss}\left(C^{[g]}\right)$

{{</admonition>}}

**Generator：**
$$
\operatorname{Loss}(G)=-E_{z \sim p(z)}[\log (D(G(z)))]+E_{z \sim p(z)}\left[\operatorname{Loss_{CE}}\left(\widetilde{y}^{[g]}, C(G(z))\right)\right]
$$
第一项是对于判别器；第二项是对于分类器，分类器针对 $G$ 生成的样本希望减小 Margin，则 $G$ 与之对抗需要增大 Margin，用交叉熵损失即可。

#### Triangle GAN

Triangle GAN 可以看作是 BiGAN 或 ALI 的一种扩展：

![Triangle GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211125155752.png "Triangle GAN")

结构上多了一个 Discriminator，一个用于判别数据对是从 $(G(z),z)$ 中还是 $(x,E(x))$ 中生成，另一个用来判别样本标签对是从 $(x,y)$ 中还是 $(G(z), y)$ 中获得。

#### Structured GAN

Structured GAN 研究基于指定语义或结构的半监督条件生成问题：

![Structured GAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211125160559.png "Structured GAN")

即相对于 Triangle GAN 加了一个 Conditioned Generator：

- Condition 1：$y$ 表示指定的语义
- Condition 2：$z$ 表示其他可变因素

#### $R^3$-CGAN

[$R^3$-CGAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Regularizing_Discriminative_Capability_of_CGANs_for_Semi-Supervised_Generative_Learning_CVPR_2020_paper.pdf) 提出了一种 CutMix （随机区域替换）策略，在两种类别的样本之间进行替换（跨类别的样本和真假样本），用于正则化 $C$ 和 $D_1$ ，结构与 Triangle GAN 相似。

![$R^3$-CGAN](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211125161450.png "$R^3$-CGAN")

#### Summary

以上方法的主要区别体现在基本模块的数量和类型上，比如 Generator，Encoder，Discriminator 和 Classifier。

{{<admonition type=note title="特点">}}

* CatGAN/CCGAN：在原始GAN的基础上引入类别信息
* Improved GAN/Localized GAN/CT-GAN：考虑 local information 和一致性约束
* BiGAN/ALI：添加 Encoder 模块
* Triple GAN：添加独立的 Classifier

{{</admonition>}}

### Semi-supervised VAE

变分自动编码器（Variational AutoEncoders）使用估计的后验分布 $q(z|x)$ 来代替真实的后验分布 $p(z|x)$ 。其置信下界ELBO（Evidence lower bound）写为：
$$
\log p(x) \geq \log \mathbb{E} _{q(z \mid x)}\left[\frac{p(z) p(x \mid z)}{q(z \mid x)}\right] = \mathbb{E} _{q(z \mid x)}[\log p(z) p(x \mid z)-\log q(z \mid x)]
$$
{{<admonition type=note title="为什么隐变量模型可以用于SSL？">}}

* 这是一种引入未标记数据的自然方式
* 通过隐变量的设置，可以轻松实现分离表征（表征解耦）的能力
* 可以使用变分方法

{{</admonition>}}

#### SSVAEs

一个具有隐编码表示的基于VAE的生成[模型](https://proceedings.neurips.cc/paper/2014/file/d523773c6b194f37b938d340d5d02232-Paper.pdf)。其中介绍了3个模型：

**Latent-feature discriminative model (M1)**：

![M1](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211129215847.png "M1")

即最普通的 VAE 模型，用深度网络构建 $p_{\theta}(x|z)$ 和 $q_{\phi}(z|x)$ 使用隐变量 $z$ 表示图像特征，用于后续的分类。

**Generative semi-supervised model (M2)**：

![M2](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211129220406.png "M2")

相当于Conditional-VAE，将类别 $y$ 加入隐变量，$y$ 服从多项式分布 $p(y)=\mathop{Cat}(y|\boldsymbol{\pi})$ ，从而在生成样本时加入类别信息，生成器为 $p_{\theta}(x|y,z)$ 。对于无标签的数据，可以根据后验分布 $p_{\theta}(y|x)$ 对其标签进行推理。

**Stacked generative semi-supervised model (M1+M2)**：

![M1+M2](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211201151833.png "M1+M2")

使用 M1 中的 $z$ 作为 M2 中的生成目标 $x$ ，即用 M2 中的 $z_2$ 和 $y$ 生成 $x$ 的隐变量特征表示 $z_1$ ，再用其隐变量 $z_1$ 生成 $x$ 。
$$
p_{\theta}(x,y,z_1,z_2)=p_{\theta}(x|z_1)p_{\theta}(z_1|y,z_2)p(y)p(z_2)
$$
其中 $p_{\theta}(x|z_1)$ 和 $p_{\theta}(z_1|y,z_2)$ 用神经网络建模。

对于有标签的样本，ELBO为：
$$
\log p_{\theta}(\mathbf{x}, y) \geq \mathbb{E}_{q_{\phi}(\mathbf{z} \mid \mathbf{x}, y)}\left[\log p_{\theta}(\mathbf{x} \mid y, \mathbf{z})+\log p_{\theta}(y)+\log p(\mathbf{z})-\log q_{\phi}(\mathbf{z} \mid \mathbf{x}, y)\right]=-\mathcal{L}(\mathbf{x}, y)
$$
对于无标签的样本，ELBO为：
$$
\begin{aligned} \log p_{\theta}(\mathbf{x}) & \geq \mathbb{E}_{q_{\phi}(y, \mathbf{z} \mid \mathbf{x})}\left[\log p_{\theta}(\mathbf{x} \mid y, \mathbf{z})+\log p_{\theta}(y)+\log p(\mathbf{z})-\log q_{\phi}(y, \mathbf{z} \mid \mathbf{x})\right] \\\\ &=\sum_{y} q_{\phi}(y \mid \mathbf{x})(-\mathcal{L}(\mathbf{x}, y))+\mathcal{H}\left(q_{\phi}(y \mid \mathbf{x})\right)=-\mathcal{U}(\mathbf{x}) \end{aligned}
$$
最终整个数据集的损失函数为：
$$
\mathcal{J}=\sum_{(\mathbf{x}, y) \sim \widetilde{p}_{l}} \mathcal{L}(\mathbf{x}, y)+\sum_{\mathbf{x} \sim \widetilde{p}_{u}} \mathcal{U}(\mathbf{x})
$$

#### ADGM

[Auxiliary Deep Generative Models](http://proceedings.mlr.press/v48/maaloe16.pdf) 是 SSVAEs 的一种扩展形式，加入了辅助变量 $a$：

![Auxiliary deep generative model](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211201152955.png "Auxiliary deep generative model")

他解决的关键问题是：**$q(z|x)$ 通常被定义为 diagonal Gaussian 的分布，这限制了模型的表达能力**。

加入辅助变量 $a$ 使得 $p(x,z)$ 变为 $p(x,z,a)=p(a|x,z)p(x,z)$ ，从而让变分分布 $q(z|x)=\int q(z|a,x)p(a|x)$ 变为一个一般的非高斯分布，以应对更复杂的后验分布 $p(z|x)$ ，提升模型的推断能力。

添加了 auxiliary variable 的 lower bound 变为：
$$
\log p_{\theta} (x) \geq \mathbb{E}_{q_{\phi}(a, z \mid x)}\left[ \log \frac{p(x,z,a)}{q(z|x)} \right] = \mathbb{E}_{q_{\phi}(a, z \mid x)}\left[\log \frac{p_{\theta}(a \mid z, x) p_{\theta}(x \mid z) p(z)}{q_{\phi}(a \mid x) q_{\phi}(z \mid a, x)}\right] \equiv-\mathcal{U}_{\mathrm{AVAE}}(x)
$$
其中 $p_{\theta}(a|z,x)$ 和 $q_{\phi}(a|x)$ 用神经网络建模为 diagonal Gaussian 分布。

#### Infinite VAE

[Infinite VAE]() 提出一种 VAE 的混合模型（一个非参数化的贝叶斯方法）。

_这篇没有详细看，涉及到 Dirichlet process，Gibbs sampling 和 variational inference，以后有时间再补充_

![Infinite VAE](https://gitee.com/miraclefish/picgo/raw/master/notebookPic/20211201193316.png "Infinite VAE")

#### Disentangled VAE


