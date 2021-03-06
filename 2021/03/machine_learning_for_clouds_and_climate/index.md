# AGU专著:云和气候的机器学习


本文是AGU专著《Clouds and Climate》其中的一章：《Machine Learning for Clouds and Climate》。文章**详细的介绍了机器学习在云和气候方面的应用、当前所面临的问题及未来的发展前景**。对于了解机器学习在云和气候方面的应用研究而言是一篇很好的概述类文章。

以下是对文章的编译，编译内容有删减，仅供交流学习。如有编译不当之处，敬请指正。文末可获取原文。

---

>机器学习是构建基于快速增长的地球系统数据的云和气候模型的强大工具。本文回顾了机器学习工具，包括可解释和物理指导机器学习，并概括了如何将机器学习应用到气候系统中的云相关过程，包括辐射、微物理、对流和云检测、分类、模拟以及不确定性量化。此外，还给出了如何开始使用机器学习的简短指南以及机器学习在云和气候方面的前沿进展。


## 引言

机器学习可以从数据中学习规律，而无需显示的编程。相比于传统的算法，比如云参数化而言，需要基于人类经验针对特定任务进行显示编程。

观测设备（比如远程遥感等）和高分辨率数值模式所产生的地球系统数据达到了前所未有的程度，机器学习在地球科学领域得到了迅速发展，机器学习从业者也越来越关注气候变化相关问题。

目前，机器学习已经在后处理、统计预报和短临预报方面得到广泛应用，并且在纯数据驱动天气预测方面进行了成功的尝试。

相比于数值天气预报，由于气候科学所具有的如下特点，限制了机器学习的直接应用：

- **气候模式中的很多变量无法直接观测**，比如云凝结速率（cloud condensation rates）和辐射效应（radiative eﬀects），从而限制了大多数利用机器学习模拟数值模式的尝试；
- **在云和气候科学领域，对模型结果的理解比准确率更重要**。气候科学家会避免使用那些他们认为不可解释的方法；
- **由于缺少完美的标注数据，很难对机器学习气候模型进行基准测试**；
- 在不断变化的气候中进行长期预测是一个外推问题，这意味着**仅利用目前的气候数据训练机器学习算法可能会导致未来的（没有观测到的）气候预测出现问题**。

尽管机器学习面临诸多挑战，但是机器学习在从复杂的大量数据集中提取气候统计特征方面仍是一个非常有前景的工具，而纯物理模型很难做到这一点。

## 机器学习

为了解决云和气候所面临的挑战，在介绍了机器学习工具之后，文章对物理指导和可解释性机器学习进行了总结。

### 机器学习工具概述

为了快速定位适合当前任务的机器学习算法，文章从两个层面介绍机器学习。

第一个，基于是否有基于专家或其它信息源的外部可用信息将算法分为: **监督学习** 和 **非监督学习** 。

 **监督学习** 指的是训练集有输入输出数据对，即算法学习输入（特征）和输出（目标）的最佳映射，以最小化代价函数，让算法的预测尽可能的逼近对应的目标。

**非监督学习** 指的是没有对应的输入-输出数据对，直接从数据中提取特征。比如主成分分析（principal component analysis，PCA）、自动编码、自组织映射网络（self-organizing maps）、聚类算法和生成式学习算法。

第二个，**根据训练过程中拟合的参数量对算法进行分类**。

对于特定任务来说，应该从简单的模型开始，因为简单的模型参数少更好解释，而且训练快。比如，想要预测连续输出的**回归任务**，`线性回归`是最简单的模型，对于**分类任务**， `logistic回归` 是最简单的模型。`决策树`通过简单的条件语句以匹配输入特征和输出，然后进行训练并求平均，从而可以降低预测结果的方差。`神经网络`是基于生物神经元网络，计算架构松散，是非常强大的非线性回归和分类工具，但是如果层数太深，计算量会很大。

理论上来说，在数据量充足的情况下，神经网络之类的复杂模型能够进行任意的非线性拟合，但是这种非常强的表征能力需要付出一定代价。简单的线性回归模型通过最小二乘法可以得到最优参数，但是**复杂模型往往需要随机优化，很难得到最佳解**。

**训练集样本数量需要随着模型的参数的增加而增加，复杂模型很容易过拟合训练集，从而导致模型的泛化能力变差**。为了减轻此问题，在进行机器学习训练时，通常需要对数据集进行划分，训练集用于优化模型的参数，验证集用来检测模型是否过拟合并优化模型的超参数，而测试集用来对模型进行最终的评估。测试集必须是之前模型没有见过的样本。

**在地球科学领域，由于样本通常呈现出很高的时空相关，在分割数据集时比较困难。相比于随机分割而言，按照年进行分割更为合理，即按照时间的顺序分割数据集。**

如果简单的模型不能解决问题，那么就要应用复杂模型，比如神经网络。但是神经网络需要大量的标注数据集，尤其是在气象领域，所需要的数据集的数量，可能比已有的观测要多。

在这种情况下，需要应用到迁移学习，即将在不同但密切相关的任务上训练的模型迁移到当前任务。比如在观测不足的情况下，可以利用数值模式的数据训练神经网络，然后迁移到观测数据进行参数的微调。此外，复杂模型，比如神经网络，很难进行解释。因此，现在学界在研究如何解释这种复杂模型。



![](/img/2021/03/25/fig1.png)



### Interpretable/Explainable 机器学习

>> **译注**：**Interpretable models** 本身并不是黑箱模型，即人能够理解模型的内部机制，并且知道模型是如何进行决策的；而 **Explainable models** 是黑箱模型，但是可以通过其它技术为模型预测提供后验解释。比如，如果我们详细属性图等方法，可以用来解释神经网络。这种解释具有一定的主观性。


**interpretable models** 在设计时本身就是可解释模型，即模型先天就具有可解释性，而 **explainable models** 则是试图解释训练模型的预测结果，即可解释性为后天提供。两个框架的本质都是为了设计更为透明和可信的人工智能模型。本文主要集中在可解释AI（explainable AI，XAI），帮助解释气候科学领域的机器学习模型。

图1的 <font color='green'>绿色框</font> 标示的则是**可解释(interpretable)的机器学习算法**，比如线性回归和决策树等简单的方法，在设计时就是可解释的模型。尽管在计算机视觉领域已经成功的利用一些方法解释神经网络的神经元，但是这些方法**很难应用到气候科学中那些具有模糊边界的对象**，比如`热浪(heat wave)`、`大气河(atmospheric rivers)`。

因此，气候科学领域的`XAI`主要是通过`属性方法(attribution methods)` 等理解ML模型针对特定样本的预测。

目前神经网络中的属性方法多是应用于图像分类，即给定图像和标签，那么输入中的哪些像素对于准确预测输出是最重要的？

`属性图(attribution map)`或`热力图(heatmap)`能够标示出输入图像的哪个区域对于预测最重要。目前属性方法已经成功的应用到大气科学领域。

除了 **属性方法** 之外，其它方法也试图去解释机器学习模型。比如，`backwards optimization` ，即对于给定的输出对应的最优输入是什么；`ablation studies` 指的是从模型的架构中移除某些部分，然后重新训练模型以测试这些结构对预测的影响。

最后，在利用可解释性AI方法时，应该要注意：
* 属性方法潜在的限制；
* 为用户需求和关注的内容选择可解释性方法的重要性。



![](/img/2021/03/05/fig2.png)



### 物理指导机器学习

尽管物理过程（比如云）的机器学习模型已经具有很高的可解释性，但是由于以下两个问题导致物理不一致，从而限制了机器学习在气候科学中的影响。

- 打破了已知的物理定律，比如质量和能量守恒定律；
- 对于那些尚未见到的情况（比如极端天气事件和变量分布发生变化，比如地理位置偏移和气候变化），可能无法泛化；

为了解决此问题，不同的物理结构可以整合到机器学习模型，可以称为`物理指导的机器学习(physics-guided ML，PGML)` 或 `混合机器学习-物理模型`。 我们可以将 `PGML` 分为三类，如图1中的 <font color=blue>篮框</font> 所示。

<font color=red>注意</font>：最优方法取决于任务和所使用的数据。

**不改变机器学习算法的架构，在损失函数中加入正则项，从而引入物理限制**。但这种软限制在地球科学中通常是不足的。由于气候趋势是由地球系统的能量不平衡所驱动的，所以守恒定律必须准确地成立。

**通过改变模型的结构可以强制能量守恒**，比如在神经网络中通过物理限制层强制能量守恒。但是这种强制能量守恒的方式并不足以泛化到训练集之外的情况。

**为了解决此问题，损失函数中的正则项可以强制假设基于物理模型的动力。尽管仍不是完美的解决方法，但是这种正则方法能够通过降低可能输出的范围改善尚未见过的情况的预测**，而不像纯机器学习模型在很大的状态空间中进行预测。

如果基于损失的正则不足以提供充足的物理结构，可以使用机器学习校正物理的先验偏差或矫正物理模型的自由参数。

<font color=red>注意</font>：如果对物理结构添加了太多限制可能会降低建模的灵活性。



## 云和气候的机器学习应用

### 辐射传输

辐射传输定义为电磁辐射形式的能量传输。尽管我们在每个波谱方面的大气吸收和发散方面已经有很好的经验，但是 `line-by-line` 的辐射传输计算仍很棘手，而且大气模式依赖于不同层次的近似表示，包括：

- **预先确定谱宽上的辐射传输积分**；
- **假设垂直积分，忽略辐射的三维性质**；
- **使用粗时空分辨率计算辐射传输**。

理论上来说，机器学习可以处理上述三种近似。机器学习已经应用于处理第三种的时间近似，通过使用机器学习模拟的辐射传输模型，替换原始的计算成本大的辐射传输模块，从而加速大气模式。此外，机器学习也可以应用于**从卫星图像中反演云属性和对地面的辐射通量进行统计预测**。

### 微物理

微物理过程指的是影响云和降水粒子的小尺度（次微米到厘米）过程。微物理参数化方案建模了云和降水粒子对天气和气候的影响，目前主要面临两个挑战：

- **在无法单独模拟所有粒子的情况下，如何表示云和降水粒子的的统计效应**；
- **由于云微物理知识的关键差距，尤其是冰相过程，微物理过程仍存在不确定性**；

因为微物理过程和地球大气和水循环的很多部分都有联系，对微物理过程的过度简化表示，尤其是水成物粒子类别的单参和双参粒子分布方案。因此，**微物理过程仍然是数值天气预报和气候模拟中很大的不确定性来源**。

通过分档方法可以更准确的描述粒子的分布，从而改善模拟的准确率。尽管分档方案的计算量很大，但是可以给机器学习算法提供更准确的训练集，利用机器学习模型代替计算成本大的模块。

**尽管目前有不少机器学习在微物理方面的研究，但是基于神经网络的微物理过程的模拟和传统的微物理方案的基准模拟结果仍然存在差距，无法很好的匹配**。这些限制表明：用于构建微物理过程的方程可能存在缺陷，进一步强调了微物理过程不确定性的重要性。为了量化模型预测的不确定性，一些研究者开始利用贝叶斯网络对微物理过程进行建模。


### 对流

大气对流指的是空气密度差所导致的运动。由于其多尺度特征，很难准确模拟大气对流。由于云在所有尺度上都存在辐射效应，从地面到大气对流可以垂直传输热量和水汽，在大气模式中云和对流的错误表示会导致能量平衡存在很大的误差，从而成为长期的气候预测方面最大的不确定性来源。

尽管对流解析尺度的大气模拟能够减轻此偏差，但是由于缺乏计算资源无法进行对流解析尺度的气候预测。统计算法，包括机器学习算法，目前不仅应用于加速计算成本高的模块，还可以发现新的湍流闭包（turbulent closures）。

![](/img/2021/03/05/fig3.png)



设计数据驱动次网格闭包（subgrid closures）的第一步是创建数据集。第二步则是选择机器学习算法。虽然神经网络常用于模拟次网络参数化，但是基于随机森林的模型能够得到更加符合线性物理限制的有界预测。这就是为什么随机森林和气候模式耦合之后非常稳定的原因。近年来，已有一些研究者在探索机器学习在次网格热动力方面的研究。


### 降尺度

业务气象预报和气候预报通常需要局地（如1公里）尺度的变量，但由于全球大气模式不适用于区域尺度的预报，其输出通常是较粗（如50-200公里）尺度的变量。

从机器学习的角度来说，降尺度类似超分辨率（super-resolution），目的是为了从低分辨率的数据得到高分辨率的结果。尽管CNN已经在成功应用到降水预测、卫星图像等方面的超分辨率任务。尽管战胜了双线性插值的基准方法，但是由于CNN是最小化每个像素的误差，从而导致CNN倾向于预测所有可能解的平均值，最终对极端情况造成了低估。为了解决此问题，目前已有一些研究在利用 **Relevance Vector Machine** 和 **GAN** 等算法进行超分辨率降尺度方面的研究。

### 气候分析和理解

机器学习在云和气候方面的应用主要是为了改善我们对气候的理解。机器学习映射局地时间和空间尺度上捕捉云和精细化尺度湍流对大尺度热动力的影响的能力，表明：在不考虑小尺度随机性、对流尺度组织性和对流记忆的情况下，可以近似地封闭大尺度湿热力学方程。

此外，现代机器学习可以直接应用于分析模式相关和理解信号可检测性。在这方面已经有很多研究，比如 `ENSO` 的可预测性的研究等。


![](/img/2021/03/05/fig4.png)



### 不确定性量化

不确定性主要分为以下四个方面：

- `观测误差(Observational)`：由于设备和表示误差导致；
- `结构误差(Structural)`：由于不正确的模型结构导致；
- `参数误差(Parametric)`：由于不正确的模型参数导致；
- `随机误差(Stochastic)`：由于气候本身的可变性或流体的混沌性质导致；

**观测、结构和参数误差是应该尽可能减少的，此外，应该尽可能增加模拟的精确度以保证能够复现随机误差**。很多机器学习算法，包括神经网络，都是确定性模型，无法描述模型的不确定性。相比之下，不确定性是资料同化的核心。尽管贝叶斯方法可以在给定初始先验和观测不确定性的情况可以推断状态或参数的后验分布。然而，贝叶斯方法并不总是能够处理结构和随机误差。

为了解决模型不确定性量化的问题，可以采用生成模型，比如变分自编码器、生成对抗网络等，构建概率模型，从而量化模型预测的不确定性。


### 云的分类和检测

从卫星和地基观测检测和分类云的能力是非常重要的应用，比如灾害天气的短临预报、检测不同类型云的发生以改善对气候的理解。

不同云类型和模式的检测对于气候变化研究者来说是非常重要的，因为不同的云结构对地球能量平衡具有不同的影响。

目前已经有很多关于机器学习利用不同资料在云检测和分类方面的应用。这些应用最常见的挑战是需要大量标注的数据集。通常通过如下四种方式解决：

- **手动标注**；
- **通过其它方法（比如传统的算法）生成标签**；
- **使用迁移学习降低样本量的需求**；
- **使用无监督学习**。

目前已有研究者利用无监督神经网络方法创建数据集的研究得到了比较合理的结果。然而，目前尚不清楚获得的云类别究竟代表什么，或者一个训练稍有不同的网络是否会产生类似的分类。

总的来说，**基于神经网络的方法可能会得到一些未知的模式，但是这些模式并没有得到充分的探索，而且相应的云类别结果也并不稳定**。


![](/img/2021/03/05/fig5.png)




## 机器学习工作流程

图6清晰的定义了机器学习算法要解决的任务和训练所需要的数据。对于没有数据集的任务，**首先就是需要创建数据集**。目前主要是通过人工标注、传统算法标注或者是众包等方式创建数据集。**在数据集创建完成后，还需要对数据集进行分割，从而区分训练集、验证集和测试集，确保用于评估机器学习算法性能的数据集是机器学习模型之前没有见过的**。

![](/img/2021/03/05/fig6.png)



数据准备好之后，**下一步是训练机器学习模型**。**在训练机器学习模型之前，应该确定基准模型，用于和机器学习模型进行对比**。

注意，**针对不同的任务需要选择不同的机器学习算法**。比如：对于空间结构数据，通常采用CNN；对于时间序列数据，可以选择RNN。另一个问题是需要注意到要解决的任务所使用的样本量，这决定了所选择的算法的复杂性。

最后，**可视化和分析模型的预测结果，从而检查是否存在异常预测**。特征重要性和其它方法可以帮助理解算法的内部工作机制，从而确定算法的内部缺点，以进一步改善算法。如果检测到了不真实的物理现象，通过整合物理限制到机器学习模型中可能是有效的，尤其是当数据有限或算法无法应用到未见过的气候场景时。

完整的构建上述机器学习工作流程可能需要花费数月的时间。在此过程中，通过代码版本控制工具以及频繁的创建检查点有助于再现工作流，并计算模型的构建和后续模型的改善。




## 前景

机器学习在云和气候方面的探索尚出于初始阶段，仍有很多涉及 **观测(observations)**、**建模(modeling)** 和 **理解(understanding)** 等方面的尚未探索的前沿研究。

**对于观测而言，主要挑战是缺乏观测样本**。传统的机器学习技术可以减轻此问题，比如数据增广(data augmentation)，比如图像的旋转、变化和镜像等操作可以增加样本量。或者**利用数值模式等工具生成数据训练模型**，然后利用 **迁移学习** 在稀疏观测上微调机器学习模型。或者使用 **元学习（meta-learning）** 利用少量的样本调整机器学习算法的超参数。相比于动力模式而言，机器学习模型简化了数学表示，更容易和资料同化框架集成，为业务模式的机器学习偏差校正带来了新的机会。

对于建模而言，机器学习模拟的是微物理和次网格参数化等方程未知的过程，比如气候模式中的云和替代很难离散的已知方程或拟合可解析过程的模拟。随着全球模式分辨率的提高，机器学习方法可以很容易的扩展到尺度很小的次网格参数化过程。尽管已经取得了一定的进展，但是机器学习对气候模式的模拟仍然面临尚一些的挑战，比如耦合到气候模式中的稳定性以及对不同气候的泛化表示能力。由于基础软件架构的限制，目前尚未尝试大规模的多 `GPU/TPU` 的机器学习高性能计算。

目前，**利用现代机器学习工具基于大量数据集改善我们的对气候的理解仍是一块处女地**。近来，在简单设置情况下，数据驱动方程发现工具在物理和海洋领域展示出很有前景的初步结果。但是，偏微分方程发现工具尚未应用到地球科学领域的云过程和气候模式。

**XAI有助于将模拟的成功转化为对气候的理解，但它并不能提取发生特定现象的原因，从而推动因果研究，以提高我们对气候系统的理论理解**。尽管因果发现方法在地球科学领域已经得到了成功的应用，但是在云相关过程的分析中仍待进一步探索。最后，因为生成模型可以同时进行降维和预测，可进一步探索潜在生成空间可能揭示观测和模拟数据中新的可预测性来源。


