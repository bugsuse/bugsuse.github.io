# WRF-DART同化系统运行手册


这一篇将详细记录WRF-DART同化系统从安装到测试运行的每一步，并记录其中可能出现的问题（因为之前已经运行过一次了，其中的脚本已经改动过，不再重新测试）。DART官方的运行测试手册真的是一言难尽。



运行测试环境：WRF 4.1.2 和 DART Manhattan。



### 设置

按照官方提供的说明，运行测试DART同化系统，需要安装WPS、[WRF](http://www2.mmm.ucar.edu/wrf/users/download/get_source.html)、[WRFDA](http://www2.mmm.ucar.edu/wrf/users/wrfda/download/get_source.html)和[DART](https://www2.cisl.ucar.edu/software/dart/download)，一些依赖库就不多说了。画图需要的软件有：nco和ncl。

WRFDA主要是为了**生成初始扰动集合文件和扰动边界条件文件**。如果你对WRF模式不是很熟悉，可以先看一下这个[在线手册](http://www2.mmm.ucar.edu/wrf/OnLineTutorial/index.htm)。



首先确保你有足够的存储运行DART进行集合同化的运行测试。

* 创建工作目录，假设目录名为`WORKDIR`

* 然后下载[测试文件](https://www.image.ucar.edu/wrfdart/tutorial/wrf_dart_tutorial_23May2018_v2.tar.gz)，大约15G

  ```bash
  cd WORKDIR
  wget http://www.image.ucar.edu/wrfdart/tutorial/wrf_dart_tutorial_23May2018_v2.tar.gz
  tar -xzvf wrf_dart_tutorial_23May2018_v2.tar.gz
  ```

* 下载完成之后，应该包括`icbc`、`obs_diag`、`rundir`、`scripts`、`output`、`perts` 和 `template`。注意目录名是大小写敏感的。



安装并复制文件到文件中：

* 构建DART可执行文件，这部分就不说了，之前已经说过了

* 复制 `$DART/models/wrf/shell_scripts` 到 `WORKDIR/scripts`

* 复制`WRF` 和 `WRFDA` 可执行文件和依赖文件到 `WORKDIR/rundir/WRF_RUN/`

  * 构建串行版WRF（这部分感觉并行版的也可以），复制 `real.exe` 到 `WORKDIR/rundir/WRF_RUN/real.serial.exe`
  * 复制 `da_wrfvar.exe` 到 `WORKDIR/rundir/WRF_RUN/da_wrfvar.exe`
  * 复制 `WRFDA/var/run/be.dat.cv3` 到 `WORKDIR/rundir/WRF_RUN/be.dat`

* 复制DART可执行文件到 `rundir`

  * 包括以下可执行文件：[advance_time](https://www.image.ucar.edu/DAReS/DART/Manhattan/assimilation_code/programs/advance_time/advance_time.html)、[filter](https://www.image.ucar.edu/DAReS/DART/Manhattan/assimilation_code/programs/filter/filter.html)、pert_wrf_bc、[obs_diag](https://www.image.ucar.edu/DAReS/DART/Manhattan/assimilation_code/programs/obs_diag/threed_sphere/obs_diag.html)、[obs_sequence_diag](https://www.image.ucar.edu/DAReS/DART/Manhattan/assimilation_code/programs/obs_sequence_tool/obs_sequence_tool.html)、[obs_seq_to_netcdf](https://www.image.ucar.edu/DAReS/DART/Manhattan/assimilation_code/programs/obs_seq_to_netcdf/obs_seq_to_netcdf.html)、[wrf_dart_obs_preprocess](https://www.image.ucar.edu/DAReS/DART/Manhattan/models/wrf/WRF_DART_utilities/wrf_dart_obs_preprocess.html)

  * 官方教程里指出，需要拷贝 `$DART/assimilation_code/programs/gen_sampling_err_table/work/sampling_error_correction_table.nc` 到 `rundir` 

  * 然后创建 `DART/WRF` 输入和重新启动的文件列表，文件名包括在`input_list_d01.txt` 和 `output_list_d01.txt` 中，如果不存在可以使用以下脚本生成，在`rundir` 目录下运行

    ```bash
    #!/bin/csh
    
    set num_ens = 50 
    set input_file_name  = "input_list_d01.txt" 
    set input_file_path  = "./advance_temp" 
    
    set output_file_name = "output_list_d01.txt" 
    
    set n = 1 
    
    if ( -e $input_file_name )  rm $input_file_name 
    if ( -e $output_file_name ) rm $output_file_name 
    
    while ($n <= $num_ens)
    
    set     ensstring = `printf %04d $n`
    set  in_file_name = ${input_file_path}${n}"/wrfinput_d01" 
    set out_file_name = "filter_restart_d01."$ensstring 
    
    echo $in_file_name  >> $input_file_name
    echo $out_file_name >> $output_file_name
    
    @ n++
    end
    ```



完成上述步骤之后，应该已经包含了以下文件：



|      executables:       | advance_time, filter, obs_diag, obs_seq_to_netcdf, obs_sequence_tool, pert_wrf_bc, wrf_dart_obs_preprocess |
| :---------------------: | ------------------------------------------------------------ |
|        scripts:         | new_advance_model.csh, add_bank_perts.ncl                    |
|      directories:       | WRFIN (empty), WRFOUT (empty), WRF_RUN (wrf executables and support files, except namelist.input) |
| support data and files: | input_list_d01.txt, output_list_d01.txt, sampling_error_correction_table.nc |
|       namelists:        | input.nml, namelist.input                                    |



在 `scripts` 目录下应该包含了以下脚本：

```bash
 assim_advance.csh         
 assimilate.csh
 diagnostics_obs.csh
 driver.csh                # 运行分析循环系统的主脚本
 first_advance.csh
 gen_retro_icbc.csh        # 用于生成模版文件和边界条件
 init_ensemble_var.csh     # 用于生成初始集合以运行循环资料同化
 param.csh                 # 参数配置脚本，包括系统的运行路径和参数设置
 prep_ic.csh
```

其他没有标注的脚本都是在循环同化期间需要运行的脚本。



下一步，就是生成扰动文件，可以查看 `$DART/models/wrf/shell_scripts/gen_pert_bank.csh` 脚本获取更多信息。如果使用官方的手册的话，官方提供了示例，可以解压到 `perts` 目录下。



`icbc` 目录下包含了 `geo_em_d01.nc` 和 grib 文件，可用于生成初始和边界条件文件。



`template` 目录下包含了`WPS`、`WRF` 和 `filter` 的 `namelist` 文件，以及对应的 `wrfinput` 文件。



最后，`output` 目录下包含了观测。模板文件一旦创建（将在下面完成）将放置在此处，并且随着循环的进行，输出都将放在此目录下。



### 初始条件

设置好集合成员的参数之后， 可以初始化集合成员文件，当然也可以从全球的状态集合中初始化集合。这里，我们利用随机误差构建了一系列流依赖误差，然后进行简短的预报。



首先需要生成一系列GFS状态文件和边界条件文件，用于循环同化。使用`gen_retro_icbc.csh` 脚本创建初始文件，然后移动到 `output` 目录的对应日期目录下。当然，也可以放到其他目录下 ，但是需要编辑对应的 `params.csh` 中的参数。脚本运行完成后，应该会生成如下文件：

```bash
   wrfbdy_d01_152057_21600_mean
   obs_seq.out
   wrfinput_d01_152057_21600_mean
   wrfinput_d01_152057_0_mean
```



下一步，运行脚本生成初始集合用于首次分析。`init_ensemble_var.csh`  此脚本默认会生成50个小脚本，然后提交运行。运行完成之后，在`ouutput/日期/PRIORS` 对应的目录下会生成类似 `prior_d01.0001, prior_d01.0002` 等文件。



> 官方提供的脚本中使用的是两种任务管理系统，可能需要根据你的运行环境修改脚本。



### 准备观测

官方教程中提供了观测序列文件让你更快的运行测试系统。观测处理对于成功的获取结果来说是非常关键的，一定要花时间把这部分弄清楚。



DART提供了一系列工具转换标准的观测格式为DART使用的[观测序列文件](https://www.image.ucar.edu/DAReS/DART/DART2_Observations.php#obs_seq_overview)。详细信息可以参考DART的[观测文档](https://www.image.ucar.edu/DAReS/DART/Manhattan/observations/obs_converters/observations.html)。



为了控制要转换的观测值，可以通过如下的 `namelist` 控制。在input.nml中，可以设置如下参数转换 `bufr`：

```bash
&prep_bufr_nml
   obs_window    = 1.0
   obs_window_cw = 1.5
   otype_use     = 120.0, 130.0, 131.0, 132.0, 133.0, 180.0
                   181.0, 182.0, 220.0, 221.0, 230.0, 231.0
                   232.0, 233.0, 242.0, 243.0, 245.0, 246.0
                   252.0, 253.0, 255.0, 280.0, 281.0, 282.0
   qctype_use    = 0,1,2,3,15
   /
```



上述参数定义了滑动窗口为+/- 1小时，而云的移动时间窗口为 +/- 1.5小时。使用的观测类型包括sounding temps (120), aircraft temps (130,131), dropsonde temps (132), mdcars aircraft temps, marine temp (180), land humidity (181), ship humidity (182), rawinsonde U,V (220), pibal U,V (221), Aircraft U,V (230,231,232), cloudsat winds (242,243,245), GOES water vapor (246), sat winds (252,253,255), ship obs (280, 281, 282)。而且仅包括指定质控类型的观测。在 [prebufr](https://www.image.ucar.edu/DAReS/DART/Manhattan/observations/obs_converters/NCEP/prep_bufr/prep_bufr.html) 可以获取更多信息。可以复制上述参数的 `input.nml` 到 `DART/observations/obs_converters/NCEP/prep_bufr/work/` 目录。



在 `DART/observations/obs_converters/NCEP/prep_bufr/work/prepbufr.csh` 脚本中包含如下信息：

```bash
set daily    = no
set zeroZ    = no # to create 06,12,18,24 convention files
set convert  = no
set block    = no
set year     = 2008
set month    = 5 # no leading zero
set beginday = 22
set endday   = 24
set BUFR_dir = ../data
```



运行shell脚本生成中间格式的txt文件。下一步，编辑 `input.nml` 文件，添加如下参数，然后复制到 `$DART/observations/NCEP/ascii_to_obs/work/` 目录，然后运行 `quickbuild.csh` 脚本。



```bash
&ncepobs_nml 
   year       = 2008 
   month      = 5 
   day        = 22 
   tot_days   = 31 
   max_num    = 800000 
   select_obs = 0 
   ObsBase    = '../../path/to/temp_obs.' 
   ADPUPA     = .true. 
   AIRCFT     = .true. 
   SATWND     = .true. 
   obs_U      = .true. 
   obs_V      = .true. 
   obs_T      = .true. 
   obs_PS     = .false. 
   obs_QV     = .false. 
   daily_file = .false. 
   lon1       = 270.0 
   lon2       = 330.0 
   lat1       = 15.0 
   lat2       = 60.0
   /
```



查看 [creat_real_obs](https://www.image.ucar.edu/DAReS/DART/Manhattan/observations/obs_converters/NCEP/ascii_to_obs/create_real_obs.html) 获取更多信息，设置和添加更多的namelist选项。运行 `create_real_obs` 可以生成一些观测序列文件，每6小时一个。对于循环试验来说，典型的做法是每个分析过程，在单独的文件中放置一个观测文件。比如在 `output` 目录下，我们创建了类似 `2012061500, 2012061506, 2012061512` 等的目录；然后将观测放到对应的目录下，比如`obs_seq2012061500`，并重命名为 `obs_seq.out`。



[wrf_dart_obs_preprocess](https://www.image.ucar.edu/DAReS/DART/Manhattan/models/wrf/WRF_DART_utilities/wrf_dart_obs_preprocess.html) 也是很有帮助的，该程序可以去除不在模拟域中的观察，可以对密集观测执行超级观测、增加侧边界附近的观测误差、检查远离模型地形高度的地面观测等。这些操作可以改善系统的性能并简化观测空间诊断的解释。有需要namelist选项可以设置，而且必须提供 wrfinput 文件给程序，以获取分析区域的信息。



### 循环分析系统

完成上述操作之后，下一步就是运行循环分析系统。对于这一步，通常建议在超算集群上使用作业管理系统进行运行。



在 `scripts` 目录下，可以发现 `param.csh` 参数设置脚本、`driver.csh` 驱动脚本、`assim_advance.csh`高级集合成员和 `assimilate.csh` 滤波等模版脚本。可以编辑参数设置脚本设置路径参数，也可以调整循环频率、模拟域、集合大小等参数；然后修改驱动脚本中的路径参数，并执行如下命令即可运行：

```bash
csh driver.csh 2017042706 >& run.out & 
```



此脚本会检查输入文件是否存在，比如wrf的边界条件、初始条件、观测序列和DART重启文件，然后创建脚本运行滤波过程，监测滤波的输出等。此过程完成后，会检测是否是最后一次分析，以确定是否需要启动新的分析。驱动脚本也会运行其他脚本以计算观测空间诊断，并转换最终的观测序列文件为nc格式。



### 检查结果

一旦分析系统运行完之后，需要检查运行的情况以确定是否存在问题。DART提供了状态和观测空间的分析系统诊断。



可以在滤波完成后，检查 `output/$date` 目录下 `analysis_increment.nc` 文件中从背景场到分析场集合平均状态的变化。也可以使用[obs_diag](https://www.image.ucar.edu/DAReS/DART/Manhattan/assimilation_code/programs/obs_diag/threed_sphere/obs_diag.html) 检查观测空间的分析统计，可以在 `output/$date/obs_diag_output.nc` 目录下找到分析结果。也可以使用[obs_seq_to_netcdf](https://www.image.ucar.edu/DAReS/DART/Manhattan/assimilation_code/programs/obs_seq_to_netcdf/obs_seq_to_netcdf.html) 将观测序列文件转换为 nc 格式，然后进行进一步的分析评估。



文件名类似`obs_epoch_029.nc`，文件中的数字表示最近处理的观测集合中的最大值。一旦执行了多次循环，额外的文件可用于绘制最近同化的观测的时间序列。确保同化了超过90%的可用观测。低的同化率表示在背景场分析、观测质量和确定观测误差时存在问题，必须要在解决。





### 参考链接

1. https://www.image.ucar.edu/wrfdart/tutorial/




