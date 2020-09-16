# YouTube-dl下载视频及ffmpeg使用操作


本文记录如何利用youtube-dl从油管下载相关视频的操作，以及下载完成之后进行字幕转换及字幕和视频的合并处理。



### youtube-dl

#### 安装

关于安装没啥可说的，直接使用pip进行安装即可。

```bash
pip install youtube-dl
```



#### 使用方法

* **列出所有格式的视频**

  ```bash
  youtube-dl --list-formats url
  或
  youtube-dl -F url
  ```

* **下载所有格式视频**

  ```bash
  youtube-dl --all-formats url
  ```

* **下载指定格式视频**

  从列出的所有格式视频信息中选择需要的格式进行下载

  ```bash
  youtube-dl -f 格式ID url
  ```

  或通过**扩展名**下载

  ```bash
  youtube-dl --f mp4 url
  ```

* **下载指定质量的视频**

  有些视频的质量可能达不到要求，可以使用如下命令下载最佳音视频质量的视频：

  如下命令将下载最佳视频和最佳音质的mp4格式视频

  ```bash
  youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' url
  ```

  如果想分别下载的话，可以将中间的`+`号改为`,`号。

* **只下载音频**

  以下命令指定只下载`mp3`格式音频

  ```bash
  youtube-dl -x --audio-format mp3 url
  ```

* **从播放列表下载特定视频**

  ```bash
  youtube-dl --playlist-items 10 url
  ```

  也可以指定多个视频

  ```bash
  youtube-dl --playlist-items 2,3,6,8 url
  ```

  或者通过范围指定

  ```bash
  youtube-dl --playlist-start 3 --playlist-end 10 url
  ```

* **下载多个视频**

  直接用空格分割视频链接即可

  ```bash
  youtube-dl url1 url2
  ```

  或者通过文件给定要下载的视频

  ```bash
  youtube-dl -a urls.txt
  ```

* **下载字幕**

  如果视频提供了字幕的话，可以直接使用如下参数下载字幕和视频

  ```bash
  youtube-dl --write-sub url
  ```

  如果想下载全部字幕，可以使用如下命令：

  ```bash
  youtube-dl --all-subs url
  ```

  如果只想下载字幕，不想下载视频，可使用如下命令：

  ```bash
  youtube-dl --write-sub --skip-download url
  ```

  如果没有字幕，可以使用如下命令下载自动生成的字幕

  ```bash
  youtube-dl --write-auto-sub url
  ```

* **下载带有描述、元数据、缩略图等的视频**

  ```bash
  youtube-dl --write-description --write-info-json --write-annotations --write-sub --write-thumbnail url
  ```

  

下载完成后，如果想转换字幕格式文件，或者转换视频格式，将单独的字幕和视频合并，那么可以使用`ffmpeg`命令。



### ffmpeg

#### 安装

以`Centos7`安装为例，使用`root`权限执行以下命令即可

```bash
yum install epel-release

rpm -v --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm

yum install ffmpeg ffmpeg-devel
```



#### 使用方法

* **字幕文件转换**

  ```bash
  ffmpeg -i subtitle.srt subtitle.ass
  或
  ffmpeg -i sub.vtt sub.art
  ```

* **集成字幕选择播放**

  播放时需要选择相应的字幕

  ```bash
  ffmpeg -i input.mp4 -i subtitles.srt -c:s mov_text -c:v copy -c:a copy output.mp4
  ```

* **嵌入字幕到视频文件**

  ```bash
  ffmpeg -i video.avi -vf subtitles=subtitle.srt out.avi
  ```

* **嵌入其他视频的字幕到视频中**

  ```bash
  ffmpeg -i video.mkv -vf subtitles=video.mkv out.avi
  ```

  嵌入第二个字幕

  ```bash
  ffmpeg -i video.mkv -vf subtitles=video.mkv:si=1 out.avi
  ```

* 嵌入`ass`字幕到视频

  ```bash
  ffmpeg -i video.avi -vf "ass=subtitle.ass" out.avi
  ```

  







### 参考链接

1. https://zhuanlan.zhihu.com/p/105141332

2. https://www.cnblogs.com/wpjamer/p/7392592.html
3. https://www.yaosansi.com/post/ffmpeg-burn-subtitles-into-video/
4. https://www.myfreax.com/how-to-install-ffmpeg-on-centos-7/




