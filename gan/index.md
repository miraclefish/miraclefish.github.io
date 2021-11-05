# GAN


## 1. Adversarial nets

为了从data $X$ 中学习generator的分布 $p_g$ ，首先定义一个先验的噪声变量分布 $p_{\boldsymbol{z}}(\boldsymbol{z})$，然后用 $G\left(\boldsymbol{z} ; \theta_{g}\right)$ 表示从 $z$ 到 $x$ 空间的映射，$G$ 是由多层感知机表示的可微函数。然后我们再定义一个多层感知机 $D\left(\boldsymbol{x} ; \theta_{d}\right)$ 输出一个数值，这个数值表示 $x$ 来自data的概率（而不是 $p_g$ ）。我们训练 $D$ 去最大化给训练样本和从 $G$ 中采样生成的样本分配正确标签的概率。同时训练 $G$ 去最小化 $\log (1-D(G(z)))$。

换言之，$D$ 和 $G$ 在玩一个minimax game：

$$\min_{G} \max_{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]$$

See Figure 1 for a less formal, more pedagogical explanation of the approach.

![Explanation](/images/GAN_explanation.png "Explanation")

## 2. Theoretical Results

![Training approach](/images/GAN_algorithm.png "Training approach")

### 2.1 Global Optimality of $p_{g}=p_{\text { data }}$

我们首先考虑，对于任意的generator $G$ 去优化discriminator $D$

**Proposition 1.**  对于给定的 $G$，最优的 $D$ 是

$$D_{G}^{*}(\boldsymbol{x})=\frac{p_{\text {data}}(\boldsymbol{x})}{p_{\text {data}}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}$$

注意，针对D的训练目标可以解释为最大化条件概率 $P(Y=y | \boldsymbol{x})$ 的最大似然函数。

则minimax game可以重新表示为：

$$\begin{aligned} C(G) & = \max_{D} V(G, D) \\\\ & =\mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}\left[\log \left(1-D_{G}^{*}(G(\boldsymbol{z}))\right)\right] \\\\ & = \mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right] \\\\ & = \mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log \frac{p_{\text { data }}(\boldsymbol{x})}{P_{\text { data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \frac{p_{g}(\boldsymbol{x})}{p_{\text { data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right] \end{aligned}$$

**Theorem 1.** 当且仅当$p_{g}=p_{\text { data }}$时，C(G)取得全局最小值。最小值为-log4。

$$
C(G)=-\log (4)+K L\left(p_{\text { data }} \| \frac{p_{\text { data }}+p_{g}}{2}\right)+K L\left(p_{g} \| \frac{p_{\text { data }}+p_{g}}{2}\right)
$$

$$
C(G)=-\log (4)+2 \cdot J S D\left(p_{\mathrm{data}} \| p_{g}\right)
$$

### 2.2 Convergence of Algorithm 1

**Proposition 2.** 如果G和D的能力足够强，在algorithm 1的每一步，D对于每一个给定的G给出最优解，并且$p_g$以提升如下的目标去更新参数：

$$\begin{aligned} C(G) & = \max_{D} V(G, D) \\\\  & \mathbb{E}_{\boldsymbol{x} \sim p_{\text { data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right] \end{aligned}$$

那么，$p_g$将收敛于$p_{data}$。

## 3. Conclusion

![Challenges in generative modeling](/images/GAN_conclusion.png "Challenges in generative modeling")

---
**Advantages and disadvantages**

- **Disdvantages**
    - 1. $p_g(x)$没有准确的表示
    - 2. 在训练期间，$D$必须与$G$同步（尤其是，在不更新$D$的情况下$G$不能训练太多次）

- **Advantages**
    - 1. 不需要Markov chain了，只有通过backprop获得梯度
    - 2. 学习过程不需要推理
    - 3. 可以将更广泛的函数合并到模型中
    - 4. Adversarial模型也从生成网络中获得了一些统计优势（不是用数据样本直接更新参数，而是使用流经discriminator的梯度），这意味着输入的组成不是直接复制generator的参数；另一个优势是，它可以表征非常尖锐，甚至退化的分布，而基于马尔可夫链的方法要求分布比较模糊，以便链条能够在模式之间混合

---

## Referance

> 1. [Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial networks[J]. Communications of the ACM, 2020, 63(11): 139-144.](https://dl.acm.org/doi/pdf/10.1145/3422622)
> 2. [Goodfellow I. Nips 2016 tutorial: Generative adversarial networks[J]. arXiv preprint arXiv:1701.00160, 2016.](https://dl.acm.org/doi/pdf/10.1145/3422622)
