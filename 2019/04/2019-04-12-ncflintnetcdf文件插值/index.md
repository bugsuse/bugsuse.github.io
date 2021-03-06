# ncflint:NetCDF文件插值


`ncflint`可以将输入文件进行线性组合创建输出文件。支持的线性组合方式为：**加权平均**，**归一化加权平均**，**插值**。坐标变量的处理方式和之前的命令不同，`ncflint`将直接拷贝坐标变量。

使用`ncflint`有两种方法：

### **指定每个文件的权重**

  比如，假设根据两个文件计算，输出文件中的变量var3的值，由输入文件中的变量var1和var2及所分配的权重来决定：即`var3 = var1*wgt1 + var2*wgt2`。权重系数通过`-w`或`--wgt_var`/`--weight`指定，如果只给定`wgt1`，那么`wgt2 = 1 - wgt1`。

  **注意：权重系数可以大于1。如果大于1，就是给定文件的变量直接扩大多少倍。**

  从V4.6.1开始，添加了`-N`选项(长选项为`--nrm`/`--normalize`)，这将对权重执行归一化操作，即`wgt1 = wgt1/(wgt1+wgt2)`，`wgt2=wgt2/(wgt1+wgt2)`。

  **注意：请小心使用`-N`归一化选项，因为这和`ncwa`的操作不同。**

### **执行插值选项**
  即`-i`，长选项为`--ntp`/`--interpolate`

  这与指定权重是相反的。当指定权重的时候，输入值乘上权重然后相加，不再进行其他操作，这意味着，要事先知道权重信息。

  另一类问题是：当已经知道var3时(确定想要获取到什么值)。针对这种情况，需要根据输入文件的值来推断权重。这就形成了有两个未知数的方程：`var3 = wgt1*var1 + wgt2*var2`。如果要确定权重，那么就要强制添加一个限制，即对权重进行归一化，添加`wgt1+ wgt2=1`限制。

  因此使用插值选项时，用户要使用`-i`选项指定`var`和`var3`。`ncflint`会计算权重`wgt1`和`wgt2`，然后使用这些权重来创建输出文件。尽管在输入文件中`var`可能有多个维度，但是必须要是一个标量。因此，和`var`相关的任何维度都要降维。



如果没有指定`-i`或`-w`，那么`ncflint`默认对每个输入文件使用相同的权重。即`-w 0.5`或`-w 0.5,0.5`。如果试图同时指定`-i`和`-w`选项，将会导致错误。

`ncflint`仅对数值变量进行操作，不支持`NC_CHAR`和`NC_STRING`类型变量。

默认情况下，`ncflint`仅对记录变量(record variables)进行操作(`time`通常存储为记录变量)，而不操作坐标变量(即latitude和longitude)。这是因为`ncflint`通常对输入文件进行时间插值(time-interpolate)，而很少执行空间插值。

有时候，用户可能想对除坐标变量外的整个文件乘一个常数，可以使用`--fix_rec_crd`选项执行这一操作(V4.2.6之后)。这防止`ncflint`对记录变量外的坐标变量也执行相同操作。


  > 注意：
  >
  > 当存在缺失值时，`ncflint`有时可能会出现意外的情况，比如当`var1`的某个格点为缺失值，而`var2`对应的格点不是缺失值时：
  >
  > * `var3`设置为缺失值。从本质上来说，执行插值操作时，缺失值就是未定义的值。在大多数情况下可能会导致异常，因此`ncflint`将其设置为缺失值。
  > * 输出有效权重的数据点，即`var3 = wgt2*var2`。这是因为对于多个点的加权平均而言，不应该舍弃有效值的加权结果。
  > * 返回没有加权的结果，即`var3 = var2`。
  >
  > 当前的处理策略采用的是第一种方法，即设置为缺失值。

  > 示例:
  >
  > * 假如有两个关于大气状态的文件，分别为00UTC和02UTC，想要得到01UTC时刻的大气状态，那么可以使用如下命令进行线性插值，假设每个文件中都包含`time`标量来存储时间戳信息：
  >
  > ```bash
  > ncflint -i time,1 utc00.nc utc02.nc utc01.nc
  > ```
  >
  > * 如果有两个时刻的观测数据，比如2019年1月和2019年4月，想要得到2019年2月和3月的结果，那么可以执行加权平均，1月和4月采用2:1的比例进行加权计算2月的结果：
  >
  >   ```bash
  >   ncflint -w 0.66667 201901.nc 201904.nc 201902.nc
  >   ```
  >
  > * 对文件变量乘一个常数
  >
  >   ```bash
  >   ncflint -w 3,-2 in.nc in.nc out.nc
  >   ```
  >
  > * 对排放源数据中除坐标变量意外的变量乘一个系数
  >
  >   ```bash
  >   ncflint --fix_rec_crd -w 0.8,0.0 emissions.nc emissions.nc scaled_emissions.nc
  >   ```
  >
  >   使用`--fix_rec_crd`选项时，要确保`time`坐标没有改变。
  >
  > * 对两个文件进行算数操作
  >
  >   ```bash
  >   ncflint -w 1,1 in1.nc in2.nc in1pin2.nc  # 两个文件相加
  >   ncflint -w 1,-1 in1.nc in2.nc in1min2.nc     # 两个文件相减，in1 - in2
  >   ```
  >
  >   `ncflint`可以模仿`ncbo`的一些操作，但是尽量不要这样做，因为`ncflint`不支持广播操作。
  >
  > * 大气压单位转换
  >
  >   即改变`prs_sfc`的单位，从Pa转为hPa
  >
  >   ```bash
  >   ncflint -C -v prs_sfc -w 0.01,0.0 in.nc in.nc out.nc
  >   ncatted -a units,prs_sfc,o,c,millibar out.nc
  >   ```



