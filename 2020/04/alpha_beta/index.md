# alpha-beta-gamma滤波


## 示例一：黄金重量

在此示例中，我们估计静态系统的状态。静态系统指的是随着时间的演变状态不会发生变化。比如，高塔就属于静态系统，高度是其状态。



在此示例中，我们可以评估黄金的重量。我们有一个无偏的秤，即秤的测量**不存在系统误差**，但是测量包括**随机噪声**。



此示例中，系统是黄金，系统状态是黄金的重量。系统的动态模型是常数，因此我们假设重量不会随着时间发生变化。为了估计系统的状态，即重量，我们进行了多次测量，并求均值。



<img src="/img/2020/04/18/ex1_MeasurementsVsTrueVal.png" style="zoom:75%;" />

第N次测量，估计值 $x_{N, N}$ 是之前所有测量的均值。
$$
\hat{x}_{N, N}=\frac{1}{N}\left(z_{1}+z_{2}+\ldots+z_{N-1}+z_{N}\right)=\frac{1}{N} \sum_{n=1}^{N}\left(z_{n}\right)
$$

> 注意：
>
> $x$ 表示重量的真值；
>
> $z_n$ 表示第 n 次重量的测量结果；
>
> ${\hat{x}_{n, n}}$ 表示第 n 次测量 $x$ 的估计，$z_n$ 测量后的估计；
>
> ${\hat{x}_{n, n-1}}$ 表示第 $n-1$ 的估计；$z_{n-1}$ 测量后的估计；
>
> ${\hat{x}_{n+1, n}}$  表示 $x$ 未来 $n + 1$ 状态的估计。第  $n$ 次进行的估计，在测量 $z_n$ 之后。换句话说，${\bar{x}_{n+1, n}}$ 表示预测状态。



> 注意：在文献中，带有 ^ 的变量用于表示估计值。



此例中，动态模型是不变的，即 $x_{n+1, n}=x_{n, n}$。



为了估计 ${\hat{x}_{n, n}}$ 我们需要记住所有的历史测量结果。如果我们没法记录历史测量结果，那么我们也不能依靠记忆记住所有的历史测量结果。我们可以使用之前的估计结果，并添加微小的调整。我们可以通过数学上的小技巧实现：

$$
\begin{aligned}
&\hat{x}_{N, N}=\frac{1}{N} \sum_{n=1}^{N}\left(z_{n}\right)=\\
&\begin{array}{c}
=\frac{1}{N}\left(\sum_{n=1}^{N-1}\left(z_{n}\right)+z_{N}\right)= \\
=\frac{1}{N} \sum_{n=1}^{N-1}\left(z_{n}\right)+\frac{1}{N} z N= \\
=\frac{1}{N} \frac{N-1}{N-1} \sum_{n=1}^{N-1}\left(z_{n}\right)+\frac{1}{N} z_{N}= \\
=\frac{N-1}{N} \frac{1}{N-1} \sum_{n=1}^{N-1}\left(z_{n}\right)+\frac{1}{N} z N=
\end{array}\\
&=\frac{N-1}{N} \hat{x}_{N, N-1}+\frac{1}{N} z_{N}=\\
&=\hat{x}_{N, N-1}-\frac{1}{N} \hat{x}_{N, N-1}+\frac{1}{N} z_{N}=\\
&=\hat{x}_{N, N-1}+\frac{1}{N}\left(z_{N}-\hat{x}_{N, N-1}\right)
\end{aligned}
$$

$\hat{x}_{N, N-1}$ 是基于 $N-1$ 次的结果预测的 $N$ 次 $x$ 的状态。换句话说，$\hat{x}_{N, N-1}$ 是之前的估计。上面的等式是五个卡尔曼滤波等式之一，称为**状态更新方程(**State Update Equation)**。如下：

![](/img/2020/04/18/ex1_stateUpdate.png)

在此例中，`factor` 表示 $\frac{1}{N}$ 。我们稍后会讨论 `factor` 的重要性。但是现在，我们要记住的是：在**卡尔曼滤波**中，这个`factor` 称为 **卡尔曼增益(Kalman Gain)**，通过 $K_n$ 表示。下标 $n$ 表示随着每次迭代而变化的**卡尔曼增益**。



Rudolf Kalman的主要贡献就是发现了 $K_n$ 。与此同时，在我们进一步讨论卡尔曼滤波之前，我们使用 $a_n$ 代替 $K_n$。因此，**状态更新等式**可以表示为：

$$
\hat{x}_{n, n}=\hat{x}_{n, n-1}+\alpha_{n}\left(z_{n}-\hat{x}_{n, n-1}\right)
$$


$(z_{n}-\hat{x}_{n, n-1})$ 表示测量残差，也称为**innovation**。**innovation **中包含了新的信息。



在此例中，$\frac{1}{N}$ 随着 $N$ 的增加而降低。这意味着，从一开始，我们就没有关于重量的足够信息。因此，我们基于测量进行估计。每多一次测量就会在评估中多一些重量信息，因为 $\frac{1}{N}$ 在减小。当达到一定程度时，新的测量的贡献将微乎其微。



让我们继续，在第一次测量之前，让我们简单的猜测一下黄金的重量。这就是**初猜值(Initial Guess)**，也是我们的第一次估计。



正如我们后面将看到的，卡尔曼滤波需要初猜值进行初始化，这仅是大概的猜测。



### 估计算法

下图是在此例中使用的估计算法流程图

![](/img/2020/04/18/ex1_estimationAlgorithm.png)



现在，我们开始测量并进行估计。



### 数值示例

#### 第0次迭代

* **初始化**

  初始猜测黄金的重量是 1000 g。初猜值仅在滤波初始化时使用一次，在下次迭代就不需要了。
  
  $$
  \hat{x}_{0,0}=1000 g
  $$

* **预测**

  假设黄金的重量是不会发生变化的。那么系统的动态模型就是静态的，下一次估计（预测）等于初始值。
  
  $$
  \hat{x}_{1,0}=\hat{x}_{0,0}=1000 g
  $$

#### 首次迭代

**第一步**

测量结果为

$$
z_{1}=1030 g
$$

**第二步**

计算增益，此例中 $a_n = \frac{1}{n}$ ，因此

$$
\alpha_{1}=\frac{1}{1}=1
$$

使用**状态更新方程**计算当前的估计

$$
\hat{x}_{1,1}=\hat{x}_{1,0}+\alpha_{1}\left(z_{1}-\hat{x}_{1,0}\right)=1000+1(1030-1000)=1030 g
$$


> 注意：此例中，初猜值可以是任意值。因此，增益为1，初猜值会在第一次迭代后被消除。



**第三步**

系统的动态模型是静态的。因此，黄金的重量不会发生变化。下一次的状态估计（预测）等于当前的状态估计。

$$
\hat{x}_{2,1}=\hat{x}_{1,1}=1030 g
$$

#### 第二次迭代

在单位时间之后，之前迭代的**预测估计**成为了**当前迭代**的**前一次估计**。

$$
\hat{x}_{2,1}=1030 g
$$

**第一步**

进行第二次重量测量

$$
z_{2}=989 g
$$

**第二步**

计算增益

$$
\alpha_{2}=\frac{1}{2}
$$

计算当前的估计

$$
\hat{x}_{2,2}=\hat{x}_{2,1}+\alpha_{2}\left(z_{2}-\hat{x}_{2,1}\right)=1030+\frac{1}{2}(989-1030)=1009.5 g
$$

**第三步**

$$
\hat{x}_{3,2}=\hat{x}_{2,2}=1009.5 g
$$


#### 第三次迭代

$$
\begin{array}{c}
z_{3}=1017 g \\
\hat{x}_{3,3}=1009.5+\frac{1}{3}(1017-1009.5)=1012 g \\
\hat{x}_{4,3}=1012 g
\end{array}
$$

#### 第四次迭代

$$
\begin{array}{c}
z_{4}=1009 g \\
\hat{x}_{4,4}=1012+\frac{1}{4}(1009-1012)=1011.25 g \\
\hat{x}_{5,4}=1011.25 g
\end{array}
$$

#### 第五次迭代

$$
\begin{array}{c}
z_{5}=1013 g \\
\hat{x}_{5,5}=1011.25+\frac{1}{5}(1013-1011.25)=1011.6 g \\
\hat{x}_{6,5}=1011.6 g
\end{array}
$$

#### 第十次迭代

$$
\begin{array}{c}
z_{10}=1011 g \\
\hat{x}_{10,10}=1011+\frac{1}{10}(1011-1011)=1011 g \\
\hat{x}_{11,10}=1011 g
\end{array}
$$

增益随着测量次数的增加而降低。所以，每次连续测量的贡献都低于之前测量的贡献。我们得到的和真实重量最接近的是1010 g。如果我们进行了足够多的测量，结果将于真实值更为接近。



下表总结了我们的测量和估计，表中对比了测量值、估计值和真实值。

$$
\begin{aligned}
\begin{array}{ccccccccccc}
n & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\
\hline \alpha_{n} & 1 & \frac{1}{2} & \frac{1}{3} & \frac{1}{4} & \frac{1}{5} & \frac{1}{6} & \frac{1}{7} & \frac{1}{8} & \frac{1}{9} & \frac{1}{10} \\
\hline z_{n} & 1030 & 989 & 1017 & 1009 & 1013 & 979 & 1008 & 1042 & 1012 & 1011 \\
\hat{x}_{n n} & 1030 & 1009.5 & 1012 & 1011.25 & 1011.6 & 1006.17 & 1006.43 & 1010.87 & 1011 & 1011 \\
\hline \hat{x}_{n+1 . n} & 1030 & 1009.5 & 1012 & 1011.25 & 1011.6 & 1006.17 & 1006.43 & 1010.87 & 1011 & 1011
\end{array}
\end{aligned}
$$


<img src="/img/2020/04/18/KF/ex1_MeasVsTrueVsEst.png" style="zoom:75%;" />



我们可以看到，我们的估计算法对测量有平滑效果，总是向真值收敛。



### 示例总结

此例中，我们开发了简单的静态系统的估计算法。我们也推导了**状态更新方程**。





## 示例二：在一维中追踪匀速飞机

此例中我们分析随时间变化的动态系统。我们将使用$\alpha - \beta$ 滤波在一维中追踪匀速飞行的飞机。

我们首先假设在一个一维的世界中，飞机正远离或飞向雷达，雷达角度是常数，飞机的高度也是常数。如下图所示：

![](/img/2020/04/18/ex2_oneD_radar.png)



$x_n$ 表示在时间 $n$ 时到飞机的距离。飞机的速度定义为距离的变化和相应的时间的比值，如下式所示：

$$
\dot{x}=v=\frac{d x}{d t}
$$

雷达以固定的时间周期向目标发送追踪光束，时间周期为 $\Delta{t}$。

假设速度为常数，那么通过如下两个方程可以描述系统的动态模型：

$$
\begin{aligned}
&x_{n+1}=x_{n}+\Delta t \dot{x}_{n}\\
&\dot{x}_{n+1}=\dot{x}_{n}
\end{aligned}
$$

根据上述方程，在下次追踪时飞机的距离等于当前追踪的距离加上目标速度乘追踪时间周期。此例中，我们假设速度为常数。因此，下一次循环的速度等于当前循环的速度。



上述方程表示的系统称为**状态外推方程**或**预测方程**，是五个卡尔曼滤波方程其中之一。外推方程的系统是当前的状态到下一个状态（预测）。

在之前的例子中我们使用了**状态外推方程**，即我们假设下一次状态黄金的重量等于当前的重量。

状态外推方程的一般形式可以表示为矩阵形式，稍后将进行介绍。此例中，针对我们的示例，将使用上述方程。



> 注意：我们已经学习了五个卡尔曼滤波方程中的两个。
>
> * 状态更新方程
> * 状态外推方程



下面将针对我们的示例，更改**状态更新方程**。



### $\alpha - \beta$ 滤波

雷达的追踪周期$\Delta{t}$为5 s。假设在时间 $n$ 时无人机的距离为 30000 m，速度为 40 m/s。

使用**状态外推方程**我们可以预测下一时刻 $N + 1$ 时目标位置：

$$
\hat{x}_{n+1, n}=\hat{x}_{n, n}+\Delta \hat{x} \hat{x}_{n, n}=30000+5 * 40=30200 m
$$

时间 $n$ 时目标的速度：

$$
\hat{x}_{n+1, n}=\hat{x}_{n, n}=40 m / s
$$

然而，在时间 $n$ 时雷达测量的距离是 30110 m，而不是30200 m。预测和测量距离间差了90 m。出现此问题的原因可能有两个：

* 雷达测量不准确
* 飞机速度发生了变化，新的飞机速度为 $\frac{30100 - 30000}{5} = 22 m/s$



那么哪个才是真的原因呢？

我们看一下速度的**状态更新方程**：

$$
\hat{\dot{x}}_{n, n}=\hat{x}_{n, n-1}+\beta\left(\frac{z_{n}-\hat{x}_{n, n-1}}{\Delta t}\right)
$$

因子 $\beta$ 的值取决于雷达的精度等级。假设雷达的1个标准差的精度是20 m。因此，很可能是飞机素的变化导致预测和测量差距90 m。此例中，我们将 $\beta$ 设置为较大值。如果设置 
$beta = 0.9$，那么估计的速度为：
$$
\hat{\dot{x}}_{n, n}=\hat{x}_{n, n-1}+\beta\left(\frac{z_{n}-\hat{x}_{n, n-1}}{\Delta t}\right)=40+0.9\left(\frac{30110-30200}{5}\right)=23.8 m / s
$$

**状态更新方程**类似上一个例子中推导的结果。

$$
\hat{x}_{n, n}=\hat{x}_{n, n-1}+\alpha\left(z_{n}-\hat{x}_{n, n-1}\right)
$$

和之前示例不同的是，此例中的 $\alpha$ 因子是固定的，并不是在每次迭代都会发生变化。

$\alpha$ 的大小取决于雷达的测量精度。对于高精度的雷达，可以选择更大的 $\alpha$ 值。如果 $\alpha = 1$，那么测量的距离等于估计的距离。


$$
\hat{x}_{n, n}=\hat{x}_{n, n-1}+1\left(z_{n}-\hat{x}_{n, n-1}\right)=z_{n}
$$


如果 $\alpha = 0$，那么测量就没有任何意义：


$$
\hat{x}_{n, n}=\hat{x}_{n, n-1}+0\left(z_{n}-\hat{x}_{n, n-1}\right)=x_{n, n-1}
$$


现在我们得到了追踪雷达的**状态更新方程组**，也称为 $\alpha - \beta$ **追踪更新方程** 或 $\alpha - \beta$ **追踪滤波方程**。  



> 位置状态更新方程：
>
> $$
> \hat{x}_{n, n}=\hat{x}_{n, n-1}+\alpha\left(z_{n}-\hat{x}_{n, n-1}\right)
> $$
> 
> 速度状态更新方程：
> 
> $$
> \hat{\dot{x}}_{n, n}=\hat{\dot{x}}_{n, n-1}+\beta\left(\frac{z_{n}-\hat{x}_{n, n-1}}{\Delta t}\right)
> $$



> 注意：在一些书中 $\alpha - \beta$ 滤波也称为 $g -h $ 滤波。

> 注意：此例中，我们使用距离测量推导了飞机速度$\dot{x} = \frac{\Delta{x}}{\Delta{t}}$。现代雷达可以通过多普勒效应直接测量径向速度。但是我们的目标是解释卡尔曼滤波，而不是雷达的原理和操作。因此，本文仍将根据距离测量推导速度。



### 估计算法

下图是此例中所使用的评估算法示意图：

<img src="/img/2020/04/18/ex2_estimationAlgorithm.png" style="zoom:75%;" />



和示例一不同的是，此例中增益值通过 $\alpha$ 和 $\beta$ 确定。在卡尔曼滤波中，利用每次迭代计算的卡尔曼增益替换 $\alpha$ 和 $\beta$。稍后进行讨论。



### 数值示例

假设飞机在一维世界中演着雷达的径向移动，远离或接近雷达。

$\alpha - \beta$ 滤波的参数是：

* $\alpha = 0.2$
* $\beta = 0.1$

追踪周期为 5 s。



> 此例中，为了更好的理解，我们将使用不准确的雷达和低速的无人飞机。在实际生活中，雷达的准确率更高，无人飞机的速度也更快。



#### 第0次迭代

**初始化**

$n = 0$ 时的初始条件为：

$$
\begin{aligned}
&\hat{x}_{0,0}=30000 m\\
&\hat{\dot{x}}_{0,0}=40 m / s
\end{aligned}
$$

> 注意：**追踪初始化**或**如何获取初始条件**是非常重要的主题，稍后将进行讨论。现在我们的目标是理解基本的 $\alpha - \beta$ 滤波操作。首先，假设初始条件已经给定。



**预测**

使用**状态外推方程**根据初始条件外推到第一个循环（$n=1$）。

$$
\begin{array}{c}
\hat{x}_{n+1, n}=\hat{x}_{n, n}+\Delta t \hat{\dot{x}}_{n, n} \rightarrow \hat{x}_{1,0}=\hat{x}_{0,0}+\Delta t \hat{\dot{x}}_{0,0}=30000+5 \times 40=30200 m \\
\hat{\dot{x}}_{n+1, n}=\hat{\dot{x}}_{n, n} \rightarrow \hat{\dot{x}}_{1,0}=\hat{\dot{x}}_{0,0}=40 m / s
\end{array}
$$


#### 第一次迭代

在第一次循环中，初始猜测是之前的估计值：


$$
\hat{x}_{n,n-1} = \hat{x}_{1,0} = 30200 m\\ \hat{\dot{x}}_{n,n-1} = \hat{\dot{x}}_{1,0} = 40 m/s
$$


**Step1 **

雷达测量到飞机的距离


$$
z_1 = 30110 m
$$


**Step2**

使用状态更新方程计算当前的估计


$$
\begin{array}{l}\hat{x}_{1,1}=\hat{x}_{1,0}+\alpha\left(z_{1}-\hat{x}_{1,0}\right)=30200+0.2(30110-30200)=30182 m \\\hat{\dot{x}}_{1,1}=\hat{\dot{x}}_{1,0}+\beta\left(\frac{z_{1}-\hat{x}_{1,0}}{\Delta t}\right)=40+0.1\left(\frac{30110-30200}{5}\right)=38.2 m / s\end{array}
$$


**Step3**

使用状态外推方程计算下一次状态估计


$$
\begin{array}{c}\hat{x}_{2,1}=\hat{x}_{1,1}+\Delta{t} \hat{\dot{x}}_{1,1}=30182+5 \times 38.2=30373 m \\\hat{\dot{x}}_{2,1}=\hat{\dot{x}}_{1,1}=38.2 m / s\end{array}
$$


第2-10次迭代计算过程省略。



下图是真值、测量和估计值的对比：

<img src="/img/2020/04/18/ex2_lowAlphaBeta.png" style="zoom:75%;" />



从图中可以看出：估计算法对测量有平滑效果，逐渐收敛到真值。



### 使用高 $\alpha$ 和 $\beta$

下图展示了 $\alpha = 0.8$ 和  $\beta = 0.5$ 的真值、测量和估计值的对比：

<img src="/img/2020/04/18/ex2_highAlphaBeta.png" style="zoom:75%;" />



从图中可以看出，此设置的平滑效果更差。当前状态的估计非常接近测量值，但是预测估计误差非常大。

所以我们应该选择低的 $\alpha$ 和 $\beta$ 吗？

答案是否定的。$\alpha$ 和  $\beta$ 的值取决于测量的精度。如果我们使用高精度的设备，比如激光雷达，我们应该选择高的 $\alpha$ 和 $\beta$。在情况下，滤波将快速响应目标速度的变化。另一方面，如果设备的精度较低，我们应该选择低的 $\alpha$ 和 $\beta$。这种情况下，滤波将平滑测量的不确定性（误差）。然而，滤波对目标速度变化的响应将更慢。

因为 $\alpha$ 和 $\beta$ 的计算时非常重要的主题，我们稍后将进行详细介绍。



### 示例总结

此例中，我们推导了 $\alpha - \beta$ 滤波的**状态更新方程**。我们也学习了**状态外推方程**。我们基于 $\alpha-\beta$ 滤波开发了一维动态系统的估计算法，同时解决了匀速目标的数值示例。



## 示例3 追踪一维中具有加速度的飞机

此例中，我们将使用 $\alpha-\beta$ 滤波方法追踪具有固定加速度的飞机。之前的示例中，我们追踪的飞机具有40 m/s的速度。下图展示的是目标距离和速度与时间的关系：

<img src="/img/2020/04/18/ex3_constantVelocityMovement.png" style="zoom:75%;" />



如图所示，距离函数是线性变化的。现在，我们来分析具有前15 s以 50 m/s匀速飞行，之后的35 s以 $8 m/s^2$ 的加速度加速飞行的飞机。

下图是目标距离、速度和加速度与实践的关系：

<img src="/img/2020/04/18/ex3_acceleratedMovement.png" style="zoom:75%;" />



如上图所示，前15 s中飞机速度是常数，随后线性增加。距离在前15 s线性增长，随后平方增长。

我们将使用之前介绍的 $\alpha - \beta$ 滤波追踪飞机。



### 数值示例

给定飞机在一维世界中朝雷达（或远离雷达）径向移动。

$\alpha-\beta$ 参数：

* $\alpha = 0.2$
* $\beta = 0.1$

追踪周期为 5 s。



#### 第0次迭代

**初始化**

$n = 0$ 时的初始条件为：


$$
\hat{x}_{0,0} = 30000m\\\hat{\dot{x}}_{0,0} = 50 m/s
$$


> 追踪初始化是非常重要的主题，稍后将进行讨论。现在的目标是理解 $\alpha-\beta$ 滤波操作。因此，假设初始条件由其他系统给定。



**预测**

使用**状态外推方程**外推初始猜测到第一个循环（$n=1$）:


$$
\begin{array}{c}\hat{x}_{n+1, n}=\hat{x}_{n, n}+\Delta{t} \hat{\dot{x}}_{n, n} \rightarrow \hat{x}_{1,0}=\hat{x}_{0,0}+\Delta{t} \hat{\dot{x}}_{0,0}=30000+5 \times 50=30250 \mathrm{m} \\\hat{\dot{x}}_{n+1, n}=\hat{\dot{x}}_{n, n} \rightarrow \hat{\dot{x}}_{1,0}=\hat{\dot{x}}_{0,0}=50 \mathrm{m} / \mathrm{s}\end{array}
$$


第1-10次迭代过程省略。



下图比较了前75秒的真实值，测量值以及范围和速度的估计值。

<img src="/img/2020/04/18/ex3_RangeVsTime.png" style="zoom:75%;" />



<img src="/img/2020/04/18/ex3_VelocityVsTime.png" style="zoom:75%;" />



上图中可以看出真值或测量值与估计值之间存在固定的差距，称为**滞后误差(lag error)**。滞后错误的其他常用名称是：

* 动态误差（Dynamic error）
* 系统误差（Systematic error）
* 偏差误差（Bias error）
* 截断误差（Truncation error）



#### 示例总结

在此例中，我们分析了由恒定加速度引起的滞后误差。





## 使用 $\alpha-\beta-\gamma$ 滤波追踪加速飞机

此例中，我们使用 $\alpha-\beta-\gamma$ 滤波追踪具有固定加速度的飞机。



### $\alpha-\beta-\gamma$ 滤波

考虑目标还速度的 $\alpha-\beta-\gamma$ 滤波有时也称为 $g-h-k$ 滤波。因此，系统外推状态方程为：


$$
\begin{array}{c}\hat{x}_{n+1, n}=\hat{x}_{n, n}+\hat{\dot{x}}_{n, n} \Delta t+\hat{\ddot{x}}_{n, n} \frac{\Delta t^{2}}{2} \\\hat{\dot{x}}_{n+1, n}=\hat{\dot{x}}_{n, n}+\hat{\ddot{x}}_{n, n} \Delta t \\\hat{\ddot{x}}_{n+1, n}=\hat{\ddot{x}}_{n, n}\end{array}
$$


其中 $\hat{\ddot{x}}$ 表示加速度。



状态更新方程是：


$$
\begin{aligned}&\hat{x}_{n, n}=\hat{x}_{n, n-1}+\alpha\left(z_{n}-\hat{x}_{n, n-1}\right)\\&\hat{\dot{x}}_{n, n}=\hat{\dot{x}}_{n, n-1}+\beta\left(\frac{z_{n}-\hat{x}_{n, n-1}}{\Delta t}\right)\\&\hat{\ddot{x}}_{n, n}=\hat{\ddot{x}}_{n, n-1}+\gamma\left(\frac{z_{n}-\hat{x}_{n, n-1}}{0.5 \Delta t^{2}}\right)\end{aligned}
$$


### 数值示例

让我们以前面的示例为例：这架飞机以50m / s的恒定速度运动了15秒钟。然后，飞机以 $8 m/s^2$ 的恒定加速度再加速35秒。



$\alpha-\beta$ 滤波参数是：

* $\alpha = 0.5$
* $\beta = 0.4$
* $\gamma = 0.1$

追踪周期是 5 s。



> 注意：在此示例中，我们将使用非常不精确的雷达和低速目标（UAV）以获得更好的图形表示。在现实生活中，雷达通常更精确，目标可以更快。



#### 第0次迭代

**初始化**

$n = 0$时的初始条件为：
$$
\hat{x}_{0,0} = 30000m\\\hat{\dot{x}}_{0,0} = 50 m/s \\\hat{\ddot{x}}_{0,0} = 0 m/s^2
$$


**预测**

使用状态外推方程基于初始猜测外推到第一次循环（$n=1$）：


$$
\begin{aligned}\hat{x}_{n+1, n}=\hat{x}_{n, n}+\hat{\dot{x}}_{n, n} \Delta t+\hat{\ddot{x}}_{n, n} \frac{\Delta t^{2}}{2} & \rightarrow \hat{x}_{1,0}=\hat{x}_{0,0}+\hat{\dot{x}}_{0,0} \Delta t+\hat{\ddot{x}}_{0,0} \frac{\Delta t^{2}}{2}=30000+50 \times 5+0 \times \frac{5^{2}}{2}=30250 \mathrm{m} \\\hat{\dot{x}}_{n+1, n}=\hat{\dot{x}}_{n, n}+\hat{\ddot{x}}_{n, n} \Delta t & \rightarrow \hat{\dot{x}}_{1,0}=\hat{\dot{x}}_{0,0}+\hat{\ddot{x}}_{0,0} \Delta t=50+0 \times 5=50 \mathrm{m} / \mathrm{s} \\\hat{\ddot{x}}_{n+1, n} &=\hat{\ddot{x}}_{n, n} \rightarrow \hat{\ddot{x}}_{1,0}=\hat{\ddot{x}}_{0,0}=0 m / s^{2}\end{aligned}
$$


#### 第一次迭代

在第一次循环中（$n=1$），初始猜测为之前的估计：


$$
\begin{aligned}\hat{x}_{n, n-1}=& \hat{x}_{1,0}=30250 \mathrm{m} \\\hat{\dot{x}}_{n, n-1}=& \hat{\dot{x}}_{1,0}=50 \mathrm{m} / \mathrm{s} \\\hat{\ddot{x}}_{n, n-1}=& \hat{\ddot{x}}_{1,0}=0 \mathrm{m} / \mathrm{s}^{2}\end{aligned}
$$


**Step1**

雷达测量的飞机距离是：


$$
z_1 = 30160 m
$$
**Step2**

使用状态更新方程计算当前的估计：


$$
\begin{aligned}&\hat{x}_{1,1}=\hat{x}_{1,0}+\alpha\left(z_{1}-\hat{x}_{1,0}\right)=30250+0.5(30160-30250)=30205 m\\&\begin{array}{l}\hat{\dot{x}}_{1,1}=\hat{\dot{x}}_{1,0}+\beta\left(\frac{z_{1}-\hat{x}_{1,0}}{\Delta t}\right)=50+0.4\left(\frac{30160-30250}{5}\right)=42.8 \mathrm{m} / \mathrm{s} \\\hat{\ddot{x}}_{1,1}=\hat{\ddot{x}}_{1,0}+\gamma\left(\frac{z_{1}-\hat{x}_{1,0}}{0.5 \Delta t^{2}}\right)=0+0.1\left(\frac{30160-30250}{0.5 \times 5^{2}}\right)=-0.7 \mathrm{m} / \mathrm{s}^{2}\end{array}\end{aligned}
$$


**Step3**

使用**状态外推方程**计算下一次状态的估计


$$
\begin{array}{c}\hat{x}_{2,1}=\hat{x}_{1,1}+\hat{\dot{x}}_{1,1} \Delta t+\hat{\ddot{x}}_{1,1} \frac{\Delta t^{2}}{2}=30205+42.8 \times 5+(-0.7) \times \frac{5^{2}}{2}=30410 m \\\hat{\dot{x}}_{2,1}=\hat{\dot{x}}_{1,1}+\hat{\ddot{x}}_{1,1} \Delta t=42.8+(-0.7) \times 5=39.2 m / s \\\hat{\ddot{x}}_{2,1}=\hat{\ddot{x}}_{1,1}=-0.7 m / s^{2}\end{array}
$$


第2-10次迭代计算过程省略。



下图比较了前50秒的真实值，测量值和范围，速度和加速度的估计值。

<img src="/img/2020/04/18/ex4_RangeVsTime.png" style="zoom:75%;" />



<img src="/img/2020/04/18/ex4_VelocityVsTime.png" style="zoom:75%;" />



<img src="/img/2020/04/18/ex4_AccelerationVsTime.png" style="zoom:75%;" />



如上图所示， $\alpha-\beta-\gamma$ 滤波可以追踪具有恒定加速度的目标，并消除**滞后误差**。

但是在突然移动的情况下会怎么样呢？通过改变飞机方向目标会迅速发生变化。真实的目标动态模型也包括突变，即加速度变化。在这种情况下，具有固定系数的 $\alpha-\beta-\gamma$ 滤波会导致估计误差，而且在一些情况下可能会失去追踪目标。

卡尔曼滤波可以处理动态模型的不确定性，稍后将进行详细解释。



## $\alpha-\beta-(\gamma)$ 滤波总结

$\alpha-\beta-(\gamma)$ 滤波具有很多类型，而且基于相同的原理。

* 当前的状态估计基于状态更新方程；
* 下一状态的估计（预测）基于动态系统方程。



这些滤波方法间的主要差异是 $\alpha-\beta-(\gamma)$ 权重系数的选择。一些类型的方法使用常数权重系数，有一些则在每次迭代时计算权重系数。



$\alpha$、$\beta$ 和 $\gamma$ 的选择对于估计算法来说是非常关键的。另一个重要问题是**滤波的初始化**，即为第一次滤波迭代提供初始值。



以下是最常用的 $\alpha-\beta-(\gamma)$ 滤波：

* Wiener Filter
* Bayes Filter
* Fading-memory polynomial Filter
* Expanding-memory (or growing-memory) polynomial Filter
* Least-squares Filter
* Benedict–Bordner Filter
* Lumped Filter
* Discounted least-squares α−β Filter
* Critically damped α−β Filter
* Growing-memory Filter
* Kalman Filter
* Extended Kalman Filter
* Unscented Kalman Filter
* Extended Complex Kalman Filter
* Gauss-Hermite Kalman Filter
* Cubature Kalman Filter
* Particle Filter



我希望将来能写一些关于这些过滤方法的教程。但这本教程是关于卡尔曼滤波器的，这是我们下一个示例的主题。



## 参考链接

1. https://www.kalmanfilter.net/alphabeta.html






