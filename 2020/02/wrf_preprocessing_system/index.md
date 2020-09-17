# WRF模式前处理:WPS


WPS(WRF Preprocessing System)是为了真实数据模拟提供输入的前处理过程，包含了3个主要程序，分别为：`geogrid.exe`、`ungrib.exe`和`metgrid.exe`。[^1]

`geogrid.exe`

: 定义模式的模拟域，并将静态地理学数据插值到模式网格。

`ungrib.exe`

: 从GRIB格式文件中提取气象场数据。

`metgrid.exe`

: 将`ungrib.exe`提取的气象场数据**水平插值**到`geogrid.exe`定义的模拟域网格。

![](/img/2020/02/10/wps_general.png "WPS前处理流程")

如上流程图所示，WPS的各程序均通过`namelist.wps`控制文件读取相应的参数进行数据处理。但上述流程图未给出各程序(`geogrid.exe`、`ungrib.exe`和`metgrid.exe`)所需要的其他控制参数文件，比如[GEOGRID.TBL](https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Description_of_GEOGRID.TBL)、[ METGRID.TBL](https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Description_of_METGRID.TBL)和[Vtable](https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Creating_and_Editing)。

`GEOGRID.TBL`
: 定义了`geogrid.exe`插值的每个数据集的参数。每个数据集的参数是单独定义的，且通过一行`========`号进行分割。 每一部分都通过`keywords=values`形式定义。有些关键词是必须的，有些是可选的。比如：插值方法选项、平滑选项、气象场类型等。更详细的信息见文档[^2]。

`METGRID.TBL`
: 定义了`metgrid.exe`插值气象场所需要的参数。文件形式与`GEOGRID.TBL`形式类似。同样提供了插值选项，如果需要修改气象场插值选项可更改此文件插值相关参数。更详细信息见文档[^3]。

`Vtable`
: 定义了`ungrib.exe`提取气象场所需要的参数。对于常用气象数据集而言，比如NCEP，EC等机构提供的气象场，WRF中都提供了`Vtable`模版文件，一般直接链接使用即可。更多信息见文档[^4]。

: 如果你使用的是新的数据源，那么可能需要创建新的`Vtable`文件。


### WPS流程
#### geogrid
如上所述，`geogrid`主要是为了**定义模拟域**以及**插值地理数据到模式网格**。**模拟域**由`namelist.wps`中的`geogrid`部分参数定义，包含地图投影、中心经纬度以及分辨率等。`geogrid`默认会将插值`土壤类别(soil categories)`、`陆地使用类别(land use category)`、`地形高度(terrain height)`、`年平均深层土壤温度(annual mean deep soil temperature)`、`月植被覆盖率(monthly vegetation fraction)`[^5]、`月反照率(monthly albedo)`、`雪最大反照率(maximum snow albedo)`以及`坡度类别(slope category)`到模式网格。

上述所有量的全球范围数据集在WRF官方下载页均提供了下载。因为上述数据**不是时间依赖**量，因此只需要下载一次即可。有些数据集仅有一种分辨率可用，但大部分数据都提供了`full-resolution`和`low resolution`两种分辨率下载。通常`low resolution`仅作为测试和教学使用，对于其他目的应用，应使用`full-resolution`数据集。

除了默认的地理数据集，`geogrid`也能将大部分连续和类别变量插值到模拟域。如果想将新数据集插值到模拟域，可以通过修改`GEOGRID.TBL`文件实现。

运行时所需要的参数由`namelist.wps`中`&geogrid`部分提供，默认参数如下：
```bash
 &geogrid
  parent_id         =   1,   1,   2,
  parent_grid_ratio =   1,   3,   3,
  i_parent_start    =   1,  51,   39,
  j_parent_start    =   1,  31,   36,
  e_we              =  151, 154,  250,
  e_sn              =  121, 160,  283,
  geog_data_res = 'default','default','default',
  dx = 27000,
  dy = 27000,
  map_proj = 'lambert',
  ref_lat   =  40,
  ref_lon   = 116,
  truelat1  =  30.0,
  truelat2  =  60.0,
  stand_lon = 116.0,
  geog_data_path = '/public/data/geog'
 /
```

#### ungrib
`ungrib`主要负责解码GRIB格式气象场，写入**中间格式**文件。GRIB格式文件中包含了时间变化的气象场，通常这些数据来源于其他**全球/区域数值模式**，比如NCEP NAM和GFS。`ungrib`可以处理`GRIB 1`和`GRIB 2`格式文件。

处理`GRIB 2`格式文件需要编译WPS时使用`GRIB2`选项。

通常情况下，GRIB文件中包含的变量比初始化WRF模式所需要的变量要多。两种格式的文件都使用了大量编码识别GRIB文件中的变量和层。这些编码存储在`Vtable`文件中，**定义了从气象场中提取并写入到中间格式文件中的变量**。

关于上述编码的详细信息见WMO GRIB文档。

`ungrib`有三种中间格式数据可供选择：`WPS`、`SI`和`MM5`。
`WPS`
: WRF系统的一种新的格式，包含了对于下游程序非常有用的额外信息。

`SI`
: WRF系统的旧中间数据格式。

`MM5`
: 用于为MM5模式提供GRIB 2格式输入。

上述三种格式均可用于驱动WRF模式，但**推荐使用WPS格式**。

解码参数由`namelist.wps`中`&ungrib`部分提供，默认参数如下：
```bash
 &ungrib
  out_format = 'WPS',
  prefix = 'FILE',
 /
```

#### metgrid
`metgrid`将`ungrib`提取并生成的中间格式数据中的气象要素水平插值到`geogrid`定义的模拟域网格，所生成的文件可作为`real`的输入。控制水平插值的参数由`namelist.wps`中`&metgrid`部分提供，默认参数如下：

```bash
 &metgrid
  fg_name = 'FILE'
  io_form_metgrid = 2,
 /
```
关于每个气象场如何插值到模式网格，由`METGRIB.TBL`中的参数控制。可以指定插值方法、掩膜场、交错网格场(grid staggering)(ARW中是`U`和`V`，NMM中是`H`和`V`)等。


`ungrib`和`metgrid`所处理的数据均是**时间依赖**的，因此每次初始化模拟时都要运行。而对于固定区域的模拟而言，一般只需要定义一次模拟域，也仅需要一次插值静态数据到模拟域。因此`geogrid`只需要运行一次。

### 其他工具[^6]
除了上述三个程序，WPS还提供了一系列小工具，用于分析数据、可视化嵌套模拟域、计算气压场以及计算平均地面温度场。

#### avg_tsfc.exe
此程序根据给定的中间格式输入文件计算日平均地面温度。日期范围根据namelist中的`&share`部分参数设置，时间间隔为中间格式文件的时间间隔。
程序计算日均值时必须要用完整的一天的数据，如果没有完整的一天的数据则不输出。类似地，任何时间上不足一天的中间文件都会被忽略。例如，有5个6小时间隔的中间文件可用，那么最后一个文件会被忽略。
计算的平均场会以`TAVGSFC`为变量名写入到新的中间格式文件。日均地面温度场可以被`metgrid`在`namelist`中设置`constant_name`值为`TAVGSFC`进行读取。

#### mod_levs.exe

用于从中间格式文件中移除垂直层。可以通过在namelist中设置新的参数控制，比如：

```bash
&mod_levs

 press_pa = 201300 , 200100 , 100000 ,

             95000 ,  90000 ,

             85000 ,  80000 ,

             75000 ,  70000 ,

             65000 ,  60000 ,

             55000 ,  50000 ,

             45000 ,  40000 ,

             35000 ,  30000 ,

             25000 ,  20000 ,

             15000 ,  10000 ,

              5000 ,   1000

/
```

在`&mod_levs`记录中，`press_pa`变量用于控制要保留的垂直层。指定的垂直层应和中间格式文件中的`xlvl`的值相匹配。更多信息可参考[WPS中间格式](https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Writing_Meteorological_Data)的相关讨论。`mod_levs.exe`接受两个参数：第一个是输入的中间格式文件名，第二个是输出的中间格式文件名。

从气象数据集中移除垂直层是非常有用的，比如当一个数据集用作模式初始条件，另一个数据集用作模式边界条件时。对于初始条件只需要提供初始时刻的数据给`metgrid`进行插值，而边界条件则需要所有时刻的数据。如果两个数据集具有相同的垂直层，则不需要移除垂直层。由于`real`进行插值时，**需要初始条件和边界条件具有相同的垂直层**。因此，当两个数据集的垂直层不同时，则需要从`m`层的文件中移除`(m-n)`层（m>n）。m和n是这两个数据集的垂直层数。

`mod_levs`只是用于处理具有不同垂直层的不同数据集的一种折衷方法。用户在使用`mod_levs`时应该注意：尽管数据集间的垂直层位置不需要匹配，但所有数据集都应该具有**地面层数据**，而且当运行`real.exe`和`wrf.exe`时应该选择所有数据集中最低的`p_top`值。

注意：`p_top`为`namelist.input`中定义模式顶的参数。

#### calc_ecmwf_p.exe

垂直插值气象场时，`real`程序需要和其他气象场处于相同垂直层的**3D气压场**和**位势高度场**。`calc_ecmwf_p.exe`可食用ECMWF的`sigma`层数据集创建这些气象场。给定地面气压场或地面气压场的log值以及A和B的系数，那么`calc_ecmwf_p.exe`就可以计算在ECMWF `sigma` `k`层的格点 P_ijk = A_k + B_k*P_sfc_ij 的气压值。

用于ECMWF不同的垂直层数据集气压计算的系数可从以下链接获取：

```
http://www.ecmwf.int/en/forecasts/documentation-and-support/16-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/19-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/31-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/40-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/50-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/60-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/62-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/91-model-levels
http://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels
```

系数表赢写入到当前文件夹下，并命名为`ecmwf_coeffs`，以下为16层数据集的系数示例：

```
    0         0.000000      0.000000000
    1      5000.000000      0.000000000
    2      9890.519531      0.001720764
    3     14166.304688      0.013197623
    4     17346.066406      0.042217135
    5     19121.152344      0.093761623
    6     19371.250000      0.169571340
    7     18164.472656      0.268015683
    8     15742.183594      0.384274483
    9     12488.050781      0.510830879
   10      8881.824219      0.638268471
   11      5437.539063      0.756384850
   12      2626.257813      0.855612755
   13       783.296631      0.928746223
   14         0.000000      0.972985268
   15         0.000000      0.992281914
   16         0.000000      1.000000000
```

如果**土壤高度(soil height)**或**土壤位势高度(soil geopotential)**、3D温度和3D相对湿度场可用，`calc_ecmwf_p.exe`会计算3D位势高度场，这对于`real`进行准确的垂直插值是非常重要的。

给定`ungrib`解码后的中间文件和`ecmwf_oeffs`系数文件，`calc_ecmwf_p`会对namelist中给定的所有时刻数据进行循环计算，并为每个时刻生成一个中间文件，命名形式为`PRES:YYYY-MM-DD_HH`，包含所有`sigma`层的气压、位势高度以及3D相对湿度场。通过在namelist中的`fg_name`前缀列表中添加`PRES`前缀，将此中间文件和`ungrib`解码的中间文件传递给`metgrid`作为输入。


#### height_ukmo.exe
`real`对`metgrid`的输出进行垂直插值时需要3D气压场和位势高度场。而UKMO模式数据集中没有地形高度场。此程序为UKMO模式数据集计算地形高度场。

#### plotgrids.ncl
根据`namelist.wps`设置信息绘制模拟域图。在设置模拟域时非常有用，可以帮助调整namelist中关于模拟域的位置信息。可通过执行`ncl util/plotgrids.ncl`可视化模拟域。对于不同的版本的ncl，可能要调用不同的绘图脚本。对于NCL6.2及之后的版本，可使用`plotgrids_new.ncl`绘制，之前的版本可使用`plotgrids_old.ncl`。

注意：目前不支持使用`lat-lon`投影。

#### g1print.exe
输出GRIB1格式文件中的数据日期、气象场量以及垂直层等信息。

####  g2print.exe
输出GRIB2格式中数据的日期、气象场量以及垂直层等信息。可在WPS目录下执行以下命令：
```bash
./util/g2print.exe GRIBFILE.AAA
```
可得到如下信息(此处仅列出部分信息)：
```bash
 ungrib - grib edition num           2
 reading from grib file =
 GRIBFILE.AAA

      NCEP GFS Analysis
 -------------------------------------------------------------------------------
 --------
  rec Prod Cat Param  Lvl    Lvl      Lvl     Prod    Name            Time
     Fcst
  num Disc     num    code   one      two     Templ
     hour
 -------------------------------------------------------------------------------
 --------
   1   0    1  22     105       1       0       0     CLWMR    2020-02-10_00:00:00   00
   2   0    1  23     105       1       0       0     ICMR     2020-02-10_00:00:00   00
   3   0    1  24     105       1       0       0     RWMR     2020-02-10_00:00:00   00
   4   0    1  25     105       1       0       0     SNMR     2020-02-10_00:00:00   00
   5   0    1  32     105       1       0       0     GRMR     2020-02-10_00:00:00   00
   6   0   16 196      10       0       0       0     REFC     2020-02-10_00:00:00   00
   7   0   19   0       1       0       0       0     VIS      2020-02-10_00:00:00   00
   8   0    2   2     220       0       0       0     UGRD     2020-02-10_00:00:00   00
   9   0    2   3     220       0       0       0     VGRD     2020-02-10_00:00:00   00
```


#### rd_intermediate.exe
输出给定中间格式文件中的所有气象场量信息。



### 参考链接

1. https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html



<br>

[^1]: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Introduction
[^2]: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Description_of_GEOGRID.TBL
[^3]: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Description_of_METGRID.TBL
[^4]: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_Creating_and_Editing
[^5]: https://data.gov.in/keywords/vegetation-fraction-vf

[^6]: https://www2.mmm.ucar.edu/wrf/users/docs/user_guide_v4/v4.0/users_guide_chap3.html#_WPS_Utility_Programs


