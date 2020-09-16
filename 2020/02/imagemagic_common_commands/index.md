# ImageMagic实用命令


ImageMagic是用于处理图像的强大工具，鉴于有时需要处理图片，不用再专门去搜索相关内容。因此使用本文记录常用的ImageMagic命令。

### 常用命令

#### identify

查看图片的格式及大小等属性

```bash
identify input.png
```



#### convert

转换图像格式、调整图像大小、模糊、裁切、除去杂点、抖动 ( dither )、绘图、翻转、合并、重新采样等

命令格式：`command [options] input_image output_image`

* 缩放并转换图片格式

  ```bash
  convert -resize 240x180 test.png test.jpg
  ```

* 提取gif图像特定桢

  ```bash
  convert  -coalesce  'test.gif[2-5]'  test_%d.jpg
  ```

  如果gif文件包含的帧数很多，为了让帧数按照顺序排列，建议加上**前导零**，即

  ```bash
  convert  -coalesce  test.gif  frame-%03d.jpg
  ```

  * 自定义文件名[^1]

    ```bash
    convert  -coalesce  rain.gif  frame_%d.jpg
    ```

    或

    ```bash
    convert  -coalesce  -set filename:n '%p'  rain.gif  'frame_%[filename:n].jpg'
    ```

  > 转换gif为图片格式时，主要要添加`-coalesce`选项，否则提取的图片可能会出现缺失问题。
  >
  > 这是因为`-coalesce`选项会根据图像 `-dispose` 元数据的设置覆盖图像序列中的每个图像，以重现动画序列中每个点的动画效果。

* 制作gif文件

  ```bash
  convert -delay 100 'pm2.5*.png' -loop 0 pm2.5.gif
  # -delay  display the next image after pausing，即图片播放的时间间隔
  # -loop add Netscape loop extension to your GIF animation，即循环播放，0 表示无限循环
  ```

* PDF和图片间转换

  * PDF转图片

    ```bash
    convert -density 100 -background white -alpha remove test.pdf test.png
    ```

    > ImageMagic本身不支持PDF和图片的转换，需要下载额外的工具。官方推荐`ghostscript`，MacOS安装方式：`brew install ghostscript`

  * 图片转PDF

    ```bash
    convert test.png test.pdf
    ```

* 白色转透明色

  ```bash
  convert INPUT -fuzz 20% -transparent white OUTPUT.png
  ```

  或

  ```bash
  mogrify -background none -flatten test.png
  ```

  > 注意使用`mogrify`会自动覆盖原始文件。

* 添加边框

  ```bash
  convert INPUT -shave 1x1 -bordercolor black -border 1 OUTPUT
  # -bordercolor 为边框的颜色，此时表示添加黑色边框
  ```



#### montage

组合多个独立的图像来创建合成图像

* 横向合并多幅图片

  ```bash
  montage [0-5].png -tile 5x1 -geometry +0+0 out.png
  ```

  

#### composite

将多张图片合成为新的图片

* 居中叠加

  ```bash
  composite -gravity center castle.gif frame.gif castle_button.gif
  ```





### 参考链接

1. https://aotu.io/notes/2018/06/06/ImageMagick_intro/index.html
2. https://www.cnblogs.com/ittangtang/p/3951240.html

3. https://yihui.org/cn/2018/04/imagemagick/
4. https://www.hahack.com/wiki/tools-imagemagick.html



[^1]: https://aotu.io/notes/2018/06/06/ImageMagick_intro/index.html#5.2


