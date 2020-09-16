# 动手学深度学习Ch3:深度学习基础




### 过拟合和欠拟合

**过拟合(overfitting)**

: 模型的训练误差远小于它在测试数据集上的误差。

**欠拟合(underfitting)**

: 模型无法得到较低的训练误差。

造成上述问题的原因主要取决于**模型的复杂度**和**训练集数据集大小**[^1]。

![](/img/2020/02/21/capacity_vs_error.svg)

<center><font size=2 color='grey'>模型复杂度对过拟合和欠拟合的影响</font></center>



### 权重衰减

在使用`SGD`优化算法时，**权重衰减(weight decay)**等价于**L2范数正则化(L2 norm regularization)**[^2][^3]。

#### L2正则化

: 通过在模型**损失函数**后添加**惩罚项**对模型的权重参数惊醒调整，使学习到的模型参数减小。通常是应对**过拟合**的常用手段。

**L2范数惩罚项**指的是**模型权重参数每个元素的平方和**与**正常数**的乘积。即
$$
l = l_0 + \frac{\lambda}{2n}\sum_{\omega}\omega^2
$$
其中第一项$l_0$表示原来的**损失函数**，第二项表示**L2正则化项**，系数$\frac{1}{2}$是为了求导时方便。

对式(1)进行求导可得

关于$\omega$的求导
$$
\frac{\partial{l}}{\partial{\omega}} = \frac{\partial{l_0}}{\partial{\omega}} + \frac{\lambda}{n}\omega
$$
关于$b$的求导
$$
\frac{\partial{l}}{\partial{b}} = \frac{\partial{l_0}}{\partial{b}}
$$
由上述推导结果可以看出，权重衰减仅对$\omega$的更新有影响，对$b$没有影响。

$\omega$的更新方式为：


$$
\omega := \omega - \eta\frac{\partial{l_0}}{\partial{\omega}} - \frac{\eta\lambda}{n}\omega \\  := (1-\frac{\eta\lambda}{n})\omega - \eta\frac{\partial{l_0}}{\partial{\omega}}
$$
其中$\eta$表示**学习率**，$n$表示样本数。

对于**小批量随机梯度下降**而言，$\omega$和$b$的更新方式如下：


$$
\omega := (1-\frac{\eta{\lambda}}{m})\omega - \frac{\eta}{m}\sum_x\frac{\partial{l_x}}{\partial\omega}
$$

$$
b := b - \frac{\eta}{m}\sum_x\frac{\partial{l_x}}{\partial{b}}
$$


其中，$\eta$表示**学习率**，$\lambda$表示**衰减系数**，$m$表示**批量大小(mini-hatch)**。上述参数更新的后一项均变为：所有样本的导数和乘$\eta$除$m$。

关于L2正则化能够减小$\omega$，而$\omega$的减小能够防止过拟合的解释[^4]：

**奥卡姆剃刀原理解释：**更小的权重$\omega$，从某种意义上来说，意味着网络复杂度更低，对数据的拟合刚好。

过拟合情况下，模型的权重系数要尽量在每一个点都具有最小误差，那么梯度的变化就必然出现波动，梯度波动会导致梯度大小差异更大，而L2正则化的权重衰减，降低了梯度差异，从而降低了过拟合问题。



#### L1正则化

: L1正则化是在原始损失函数的基础上，添加所有权重$\omega$的绝对值之和与$\frac{\lambda}{n}$的乘积[^5]。即
$$
l = l_0 + \frac{\lambda}{n}\sum_{\omega}\omega
$$
求导可得：
$$
\frac{\partial{l}}{\partial{\omega}} = \frac{\partial{l_0}}{\partial{\omega}} + \frac{\lambda}{n}sgn(\omega)
$$
其中$sgn$表示 $\omega$ 的符号。权重$\omega$ 的更新规则变为：
$$
\omega := \omega - \frac{\eta\lambda}{n}sgn(\omega) - \eta\frac{\partial{l_0}}{\partial\omega}
$$
当$\omega$ 为正时，$sgn(\omega) = 1$，权重$\omega$减小；当$\omega$ 为负时，$sgn(\omega) = -1$，权重增大；当$\omega = 0$时，规定$sgn(\omega) = 0$。



### Dropout

**L1正则化**和**L2正则化**是通过修改模型的代价函数改善模型过拟合问题。`dropout`通过修改模型的网络结构改善模型过拟合问题。即在每一次迭代过程中，**随机**从每一个**隐藏层**中**临时删除**一半的神经元[^6]。

丢弃法不会改变输入的期望值[^7]。



### 图像增强

`torchvision.transforms`模块自带的图像增强方法所要求的输入是`PIL`图像。`transforms.ToTensor()`会将输入的尺寸为`(HxWxC)`且数据范围在`[0, 255]`间的PIL图片或数据类型为`np.uint8`的`Numpy`数组转换为尺寸为`(CxHxW)`且数据类型为`torch.float32`且位于`[0, 1.0]`的`Tensor`。



**注意**： 由于像素值为0到255的整数，所以刚好是uint8所能表示的范围，包括`transforms.ToTensor()`在内的一些关于图片的函数就默认输入的是uint8型，若不是，可能不会报错但可能得不到想要的结果。所以，**如果用像素值(0-255整数)表示图片数据，那么一律将其类型设置成uint8，避免不必要的bug。**





**更新记录**

2020.02.21 初次更新





[^1]:https://zh.d2l.ai/chapter_deep-learning-basics/underfit-overfit.html
[^2]:https://arxiv.org/pdf/1711.05101.pdf
[^3]: https://zhuanlan.zhihu.com/p/40814046
[^4]:https://zhuanlan.zhihu.com/p/58528494
[^5]: http://blog.sina.com.cn/s/blog_a89e19440102x1el.html
[^6]: https://blog.csdn.net/u012162613/article/details/44261657
[^7]: https://zh.d2l.ai/chapter_deep-learning-basics/dropout.html


