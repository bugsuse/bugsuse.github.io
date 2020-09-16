# MATLAB完美白化,支持m_map工具箱


起初的matlab完美白化脚本是发布在气象家园论坛的，但是近来气象家园论坛越发不稳定。想着逐渐迁移到个人博客。而且最初的白化脚本并不支持`m_map` 工具箱。此版本添加了`m_map`工具箱的支持。


以下是对白化函数的测试：

首先，生成随机数据：

```python
clear, clc

z = peaks(1000);
lon = [60 150];
lat = [0 60];
[LON,LAT] = meshgrid(linspace(lon(1),lon(2),1000), linspace(lat(1), lat(2),1000));
```



### lambert

```matlab
figure
m_proj('lambert', 'longitudes', lon, 'latitudes', lat, 'par', [30, 60], 'clo', 105)
m_contourf(LON, LAT, z);
m_maskmap('../data/chinamap/中国行政区_包含沿海岛屿.shp', true, 'lon', lon, 'lat', lat, 'm_map', true);

m_proj('lambert', 'longitudes', lon, 'latitudes', [15, 60], 'par', [30, 60], 'clo', 105)
m_grid('box','on', 'linestyle', 'none', 'tickdir', 'out', 'linewidth', 3);
m_mapshow('../data/chinamap/cnmap/cnhimap.shp')
```



![](/img/2020/05/01/lambert.jpg)


lambert投影的白化存在一点小问题。因为lambert投影是对原始坐标的裁切然后重新标注坐标轴，所以导致在白化时出现边缘无法完全白化的问题。


![](/img/2020/05/01/lambert_nowork.jpg)


这个问题可以通过扩大坐标轴的范围，然后重新设置投影来解决。如上述代码中所示，但注意 `par` 和 `clo` 的参数应该一致。


### mercator

```matlab
figure
m_proj('mercator', 'longitudes', lon, 'latitudes', lat)
m_contourf(LON, LAT, z);
m_maskmap('../data/chinamap/中国行政区_包含沿海岛屿.shp', true, 'lon', lon, 'lat', lat, 'm_map', true);
m_mapshow('../data/chinamap/cnmap/cnhimap.shp')
m_grid('box','on');
```

![](/img/2020/05/01/mercator.jpg)


### lat-lon

```matlab
figure
c = contourf(LON, LAT, z,'linestyle', 'none');
m_maskmap('../data/chinamap/中国行政区_包含沿海岛屿.shp', true, 'lon', lon, 'lat', lat);
mapshow('../data/chinamap/cnmap/cnhimap.shp', 'color', 'k', 'displaytype', 'line')
```

![](/img/2020/05/01/lat-lon.jpg)


`m_maskmap` 和 `m_mapshow` 函数在 [github](https://github.com/bugsuse/m_map_utils) 获取。


