# 利用TROPOMI看看疫情期间NO2排放的变化


### TROPOMI

**TROPOspheric Monitoring Instrument**(TROPOMI)是搭载在Copernicus Sentinel-5 Precursor(S5P)卫星上用于大气成分观测的仪器，于2017年10月13日发射，预计服役期为7年。

TROPOMI的目标是为了及时并准确的提供关键大气成分的观测，主要服务于空气质量、气候变化以及臭氧层监测。TROPOMI的日全球观测将用于改善空气质量预报以及大气成分浓度的监测。趋势监测对于污染排放相关的政策决策非常有用。此外，TROPOMI还可用于火山灰监测以及航空安全方面，紫外线辐射产品可用于预警强紫外辐射。

高分辨率大气成分的观测，不仅能改善空气质量数值预报，而且对于改进大气中重要化学和动力过程的理解也具有积极影响。

<img src="/img/2020/02/20/Atmosphere_composition_diagram.png" style="zoom:70%;" />

<center><font size=2 size='grey'>地球大气成分演变/循环示意图</font></center>

#### 数据产品

经过6个月的运行测试阶段后于2018年4月正式进入业务运行模式。从2018年6月中旬开始对外发布数据产品，目前提供了Level1和Level2两种数据产品。经过近一年的运行以及标定校准，数据产品空间分辨率从原先的7.2 km增加到5.6 km，并从2019年8月6日开始正式发布。对于空气质量监测来说，或许能带来更好的效果。


<img src="/img/2020/02/20/1.png" style="zoom:70%;" />

<center><font size=2 color='grey'>TROPOMI Level2数据产品</font></center>

**注**: 目前除了`Ozone profiles`和`UV`两个产品没有对外发布之外，其余产品均已对外发布。

#### 数据获取

S5P的数据是对外公开的，数据可通过官方Hub数据中心下载[^1]。用户名和密码均为：`s5pguest`

<img src="/img/2020/02/20/3.png" style="zoom:40%;" />

<center><font size=2 color='grey'>Hub数据选择页面</font></center>

目前数据中心提供了`L1B`和`L2`两种处理级别的数据产品，可根据需要选择下载。一般用户的可以选择下载`L2`的数据产品。`L2`数据产品提供了`HCHO`、`NO2`、`O3`、`CH4`、`CO`、`CLOUD`等产品。每一种产品有三种不同时间处理的数据产品可用：`NRT`、`OFFL`、`REPROCESSING`。

* `NRT(near-real-time)`：此数据的时效性最好，一般在观测到3小时左右即可获取。但是数据质量可能较差。
* `OFFL(offline)`：离线数据。数据时效性相对于`NRT`滞后一些，但是数据质量较高。
* `Reprocessing`：经过多次处理之后的数据，数据质量可能更好，但时效性最差。

`L2`级产品是根据每一条扫描轨道处理，对于扫描轨道出现重复的数据而言，还需要进一步处理。官方也提供了工具进行处理。比如由KNMI负责开发的[PyCAMA](https://dev.knmi.nl/projects/pycama)，[HARP](https://github.com/stcorp/harp)，[Satpy](https://github.com/pytroll/satpy)等。

除了利用上述工具自行处理之外，还可以通过Google Earth Engine获取处理好的Level3数据产品。下面以Google Earth Engine为例，获取TROPOMI $NO_2$数据产品，看一下此次新冠疫情对$NO_2$排放的影响。


### Google Earth Engine

关于Google Earth Engine的介绍之前也已经提到过了。这里就不多说了。我们先来看一下2019年和2020年春节期间中国东部地区$NO_2$浓度的空间变化情况。

#### 2019 VS 2020

**注**: 2020年的数据采用的是`NRT`近实时数据流，2019年的数据采用的是`OFFL`离线数据流。两种数据的数据质量可能存在一些差异。

<br>

<iframe seamless src="/img/2020/02/20/20200124/index.html" width="100%" height="500"></iframe>
<center><font size=2 color='grey'>图1 2020年春节期间中国东部NO2浓度空间分布</font></center>

仅从上图来看，无法说明$NO_2$浓度有什么问题。下面我们再来看一下2019年春节期间中国东部地区$NO_2$浓度的空间分布。
<br>

<iframe seamless src="/img/2020/02/20/20190205/index.html" width="100%" height="500"></iframe>
<center><font size=2 color='grey'>图2 2019年春节期间中国东部NO2浓度空间分布</font></center>

从上面两张图可以看出，在不考虑气象条件的情况下，$NO_2$浓度的差异是非常明显的。这很大程度上可能是由于在此次疫情期间的交通管制所导致。通过对元宵节期间以及最近几天的$NO_2$浓度的变化分析来看，由于一些企业还没有复工，而且大部人都还在家窝着，交通管制仍然还没有完全放开，所以工业和机动车的排放相对来说是很低的。这也就导致了$NO_2$浓度低于常规浓度水平。



下面看一下2019年和2020年元宵节期间中国东部地区$NO_2$浓度的空间分布情况。从图中可以看出：$NO_2$浓度的空间分布情况差异非常大。

<img src="/img/2020/02/20/4.png" style="zoom:50%;" />

<center><font size=2 color='grey'>图3 左侧为2019年2月17日-2月19日NO2的平均浓度分布，右侧为2020年2月6日-2月10日NO2的平均浓度分布</center></font>

以上关于2019年和2020年春节及元宵节期间中国东部地区$NO_2$浓度的空间分布情况对比均没有考虑同期气象条件的影响。仅作为学习参考。

#### 数据处理及可视化代码

以上$NO_2$数据产品来源于Google Earth Engine。所有代码均为python缩写。可视化采用的是`folium`，图中`colorbar`的添加使用的是`branca`，`geopandas`主要用于添加省界以及海岸线。

**注**：由于未知原因，以下代码可能需要使用代理才能够成功运行。


```python
import os
import geopandas
import folium
import branca
import ee

# 设置代理访问 Google Earth Engine
os.environ['http_proxy'] = 'http://127.0.0.1:1087'
os.environ['https_proxy'] = 'https://127.0.0.1:1087'

# 初始化 ee
ee.Initialize()

# 定义加载GEE影像数或者矢量数据方法
def add_ee_layer(self, ee_object, vis_params, name): 
    try:     
        if isinstance(ee_object, ee.Image):     
            map_id_dict = ee.Image(ee_object).getMapId(vis_params) 
            folium.raster_layers.TileLayer( 
                tiles = map_id_dict['tile_fetcher'].url_format, 
                attr = 'Google Earth Engine', 
                name = name, 
                overlay = True, 
                control = True 
            ).add_to(self) 
        elif isinstance(ee_object, ee.FeatureCollection):   
            ee_object_new = ee.Image().paint(ee_object, 0, 2) 
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params) 
            folium.raster_layers.TileLayer( 
                tiles = map_id_dict['tile_fetcher'].url_format, 
                attr = 'Google Earth Engine', 
                name = name, 
                overlay = True, 
                control = True 
            ).add_to(self) 
    except: 
        print("Could not display {}".format(name)) 

folium.Map.add_ee_layer = add_ee_layer 

no2 = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2').select('tropospheric_NO2_column_number_density').filterDate('2020-01-24', '2020-01-28')

viz = {
  'min': 0,
  'max': 0.0002,
  'palette': ['#FFFFFF', '#C2C8E1', '#9BA9DC', '#4C9DF4', '#86D1F9', '#A7E5EF', '#FEFC59', '#FD7822', '#FC131F', '#A3090E'],
  'opacity': 0.7,
}

my_map = folium.Map(location=[33, 114], zoom_start=5, height=600, width=780) 
my_map.add_ee_layer(no2.mean(), viz, 'DEM') 
colormap = branca.colormap.LinearColormap(viz['palette'], vmin=viz['min']*1e3, vmax=viz['max']*1e3)
colormap.caption = 'Tropospheric NO2 Column Number Density Jan 24-28 2020(1e-3 mol/m^2)'
colormap.add_to(my_map)
my_map.add_child(folium.LayerControl()) 

my_map.save('20200124.html')
```

图1和图2均是最终保存的`html`文件生成，可以交互式查看。图3采用的相同的方式生成，但是最终截图保存，并利用`convert`进行的图片水平拼接。

```bash
convert +append 2019021_19.png 20200206_10.png 1920.png
```

除了使用python进行可视化外，Google Earth Engine还提供了Code Editor可利用JS进行可视化。感兴趣的可以去把玩一下[^2]。

<img src="/img/2020/02/20/31.png" style="zoom:50%;" />

<center>
  <font size=2 color='grey'>Google Earth Engine Code Editor</font>
</center>

就到这里了。后面有时间的话再介绍一下如何使用[PyCAMA](https://dev.knmi.nl/projects/pycama)以及其他工具处理卫星数据的方法。



### 参考链接

1. https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_NRTI_L3_NO2
2. https://stackoverflow.com/questions/52911688/python-folium-choropleth-map-colors-incorrect/52981115#52981115
3. https://python-visualization.github.io/folium/quickstart.html#GeoJSON/TopoJSON-Overlays
4. https://ocefpaf.github.io/python4oceanographers/blog/2015/12/14/geopandas_folium/
5. https://ocefpaf.github.io/python4oceanographers/blog/2015/02/02/cartopy_folium_shapefile/
6. https://github.com/akkana/scripts/blob/master/mapping/polidistmap.py
7. https://zhuanlan.zhihu.com/p/106147610



[^1]: https://s5phub.copernicus.eu/dhus/#/home

[^2]: https://code.earthengine.google.com/?scriptPath=Examples%3ADatasets%2FCOPERNICUS_S5P_NRTI_L3_NO2


