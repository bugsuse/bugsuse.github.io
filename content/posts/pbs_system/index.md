# PBS作业管理系统




PBS作业管理系统是集群和超算上常用的作业管理系统之一，此外还有LSF和SLURM。在集群和超算使用作业管理系统对于任务的运行管理是非常方便的，可以有效的分配系统计算资源。



#### PBS常用命令

**作业管理**

* qsub 提交作业命令

  qsub 可以通过脚本提交任务，也可以交互式提交任务

  * 通过脚本提交任务

    pbs 在脚本中通常要设置以下参数，节点信息和cpu信息，运行的墙钟时间(实际运行时间)，队列名称(如果有)，任务名称，任务标准和错误输出等信息。 

    ```bash
    #!/bin/bash
    #PBS -l nodes=2:ppn=16
    #PBS -l walltime=24:00:00
    #PBS -q high
    #PBS -N Job_Name
    #PBS -oe
    ```

    参数解释：

    * nodes 和 ppn : 分别设置节点和每个节点使用的核信息，可以直接设置节点数和核数，比如上面直接指定为2，则使用两个节点，也可以指定节点，16表示每个节点使用16个核。 

      指定节点的节点名称可以通过 pbsnodes 查看可用节点信息，下图中的 cu01 和 cu02 即为节点名称， np 表示每个节点有多少个核。

      ![](https://ws1.sinaimg.cn/large/006tNbRwly1fxez9lnmecj313u0q6afj.jpg)

      如果系统没有 pbsnodes 命令可用，也可以执行 以下命令查看，因为本质上节点名称是对 ip 地址的映射，所有的映射关系都存储在 /ets/hosts 文件中

      ```bash
      vi /etc/hosts
      ```

      知道节点之后，可以指定节点运行作业:

      ```bash
      #PBS -l nodes=cu01:ppn=20+cu02:ppn=20
      ```

      这里节点之间使用 + 连接，ppn 对应的是每个节点使用多少个核，如果 ppn 超出了单个节点核的总数则会报错。

      **注意**: 这里的 ppn 可以低于20。

    * walltime: 表示墙钟时间，即作业最长运行多长时间，这里设置为24小时，具体根据作业实际运行需要设置。

      **注意**：有些集群管理系统对不同类型的作业进行了墙钟时长的限制，这个需要了解管理员配置的参数信息。

    * -q : 任务队列信息，通常用于区分不同的任务类型

    * -N：任务名称，可以任意设置，最好根据任务的内容来设置，便于区分

    * -oe：作业运行情况标准输出和标准错误输出，一般位作业名+e/o+作业号，-o 和 -e 可以单独设置，并且指定输出文件名称

      ```bash
      #PBS -o example.stdout
      #PBS -e example.stderr
      ```

  除了上述列出的参数外，还有一些参数可以设置，但一般不需要设置。比如任务所需要的内存和cpu等信息。

  完整的脚本示例 qsub_wrf.sh：

  ```bash
  #!/bin/csh -f
  
  #PBS -N WRF
  #PBS -l nodes=cu01:ppn=12+cu03:ppn=12
  #PBS -j oe
  #PBS -l walltime=24:00:00
  
  cd $PBS_O_WORKDIR
  setenv NPROCS `cat $PBS_NODEFILE | wc -l`
  
  ./run.wrf
  ```

  可以通过以下命令提交任务：

  ```bash
  qsub qsub_wrf.sh
  ```

  * 交互式提交任务

    交互式提交即是直接使用qsub指定参数提交

    ```bash
    qsub -N WRF -l nodes=cu02+cu04:ppn=20 -l walltime=24:00:00 ./run.wrf
    ```

* 查看作业信息

  * qstat -a ： 列出所有作业信息
  * qstat -n ：列出所有作业信息以及每个作业所使用的节点信息
  * qstat -q ：列出所有队列信息
  * qstat -u user_name：列出指定用户的作业信息
  * qstat -r：列出所有正在运行的作业
  * qstat -f job_id：列出指定任务id的信息
  * pestat：列出所有节点的状态

* 删除作业

  * qdel job_id：删除指定任务id的作业
  * qsig ：通过信号控制作业





**update**: 2018-11-26 更改指定多节点运行设置


参考链接：

1. https://www.jianshu.com/p/2f6c799ca147
2. https://kevinzjy.github.io/2018/08/13/180813-Server-PBS/



