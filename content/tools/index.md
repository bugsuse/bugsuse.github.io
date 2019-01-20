---
title: 
date: 2018-12-07 21:40:19
tags: 
---

<center><font size=4 color='#127EFF'>常用工具列表</font></center>
<center><font size=2 color='#FFBB47'>不断更新</font></center>



## 气象

### 模式

#### [WRF](http://www2.mmm.ucar.edu/wrf/users/)

* [WRF4G](https://meteo.unican.es/trac/wiki/WRF4G)：分布式计算系统的WRF模式运行和监控框架

* [Post-Processing](http://www2.mmm.ucar.edu/wrf/users/download/get_sources_pproc_util.html)：后处理工具


#### [WRFDA](http://www2.mmm.ucar.edu/wrf/users/wrfda/index.html)：WRF数据同化系统



#### [GSI](https://dtcenter.org/com-GSI/users/)：格点统计插值系统



#### [WRF-Chem](https://ruc.noaa.gov/wrf/WG11/)：大气化学模式



#### [CMAQ](https://www.epa.gov/cmaq)



#### [CAMx](http://www.camx.com/)：多尺度光化学模型



## 工具

### [ncregrid](http://www.pa.op.dlr.de/~PatrickJoeckel/ncregrid/)

​	2D/3D地球科学数据插值工具



### [nco](http://nco.sourceforge.net/)

​	NetCDF格式数据操作工具，可执行属性，变量等编辑操作，也可进行算术运算，插值，集合统计等等。



### [cdo](https://code.mpimet.mpg.de/projects/cdo/)

​	气候数据操作工具



### [GMT]()

* [GMT中文教程](https://docs.gmt-china.org/)




### Cheat-Sheet

* [Git](https://github.com/arslanbilal/git-cheat-sheet)

* [Vi](http://cenalulu.github.io/linux/all-vim-cheatsheat/)

  <center>经典版</center>

  ![](https://ws2.sinaimg.cn/large/006tNbRwgy1fygvls3qsqg30sg0k4tcq.gif)

  [教程在此](http://www.viemu.com/a_vi_vim_graphical_cheat_sheet_tutorial.html)



  <center>入门版</center>

  ![](https://ws1.sinaimg.cn/large/006tNbRwgy1fygvpj956yj31400u04qp.jpg)



  <center>进阶版</center>

  ![](https://ws4.sinaimg.cn/large/006tNbRwgy1fygvq6u574j312u0u0qv5.jpg)

  <center>现代版</center>

  ![](https://ws3.sinaimg.cn/large/006tNbRwgy1fygvkri2fhj31840u0dog.jpg)




### 编程语言

#### Python

* 配色库
  * [colormap](https://colormap.readthedocs.io/en/latest/): 用于颜色类型转换以及构建colormap
  * [cmocean](https://matplotlib.org/cmocean): 海洋科学绘图colormap，除python版外，还有对应的其他版本
  * [colorcet](http://colorcet.pyviz.org): 提供了很多可选的colormap
  * [palettable](https://jiffyclub.github.io/palettable): 支持很多颜色模式

* 气象相关库

  * [Metpy](https://unidata.github.io/MetPy/latest/index.html)：气象数据读取，计算和可视化工具

  * [Sharppy](https://github.com/sharppy/SHARPpy)：探空和风场分析工具

  * [Siphon](https://unidata.github.io/siphon/latest/index.html)：远程数据下载工具

  * [salem](https://salem.readthedocs.io/en/latest/)：地球科学数据处理和绘图库，其中提供了WRF模式相关函数，比如WPS绘图，垂直插值等。

  * [wrf-python](https://wrf-python.readthedocs.io/en/latest/)：WRF模式后处理库

* 地图相关库

  * [Basemap](https://matplotlib.org/basemap/)
  * [Cartopy](https://scitools.org.uk/cartopy/docs/latest/)



#### MATLAB

* [m_map](https://www.eoas.ubc.ca/~rich/map.html)：地图工具箱
* [MeteoLab](https://grupos.unican.es/ai/meteo/meteolab.html)：气象机器学习工具箱



#### [NCL](https://www.ncl.ucar.edu/)

* [NCL中文教程](https://ncl.readthedocs.io/zh_CN/latest/)


### 可视化工具

#### 配色工具

* [Colorbrewer2](http://colorbrewer2.org): 提供便捷的在线配色方案 

* [peise](http://www.peise.net/tools/web): 功能更加强大的在线配色器

#### 配色博客

* [peter](https://peterkovesi.com/projects/colourmaps)



### GIS

* [QGIS](https://qgis.org/en/site/): 多平台开源的地理信息处理软件



#### [Vis5D](http://vis5d.sourceforge.net/)

​	5D数据可视化工具，之前气象数值模式数据的可视化则是采用此工具，但目前已经被新的高维数据可视化工具代替。



#### [IDV](https://www.unidata.ucar.edu/software/idv/)

​	新一代高维数据可视化工具。