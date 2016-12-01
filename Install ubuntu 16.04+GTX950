<重要>
1.建議安裝ubuntu 16.04，因為14.04的版本太難安裝
2.先在Bias設定secure boot，將windows UEFI改成其他OS

此篇章不介紹安裝ubuntu 16.04的流程

<step1>
手動修改grub文件：

sudo vim /etc/default/grub

# The resolution used on graphical terminal
# note that you can use only modes which your graphic card supports via VBE
# you can see them in real GRUB with the command `vbeinfo’
#GRUB_GFXMODE=640×480
# 這裡分辨率自行設置
ex.
#GRUB_GFXMODE=1024×768

sudo update-grub



<step2>
安裝SSH Server，這樣可以遠程ssh操作：

sudo apt-get install openssh-server



<step3>
更新Ubuntu 16.04來源，用的是中科大的來源：

cd /etc/apt/
sudo cp sources.list sources.list.bak
sudo vi sources.list

把下面的這些加到source.list文件頂部(第一段與第二段中間空白開始貼起)：

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
剛進入此檔案會有35行，貼完下面會有63行
將游標移置第一段與第二段中間空白
按ESC，再按i
然後複製下列第一行，右鍵貼上
再按ESC，再按i
然後複製下列第二行，右鍵貼上
......以此類推操作
deb http://mirrors.ustc.edu.cn/ubuntu/ xenial main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ xenial main restricted universe multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse

全部貼完按ESC，再按:w，再按ENTER就會跳出文件並儲存

提醒：
1.假設有格式亂掉-->按ESC，再按:q!，再按ENTER就會跳出文件不會做儲存動作
2.請勿全部一起貼上去，格式一定會亂掉


最後更新已安装的套件：

sudo apt-get update
sudo apt-get upgrade



<step4>
安裝GTX950驅動
安裝 Nvidia 375
假設你不知道你的版本，可以從http://www.nvidia.com.tw/Download/index.aspx?lang=tw查詢(系統記得選Linux-64bits)
sudo add-apt-repository ppa:graphics-drivers/ppa


接著
sudo apt-get update
sudo apt-get install nvidia-375
sudo apt-get install mesa-common-dev
sudo apt-get install freeglut3-dev

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
先重新開機，驅動才會起啟動






<step5>
CUDA下載網址
https://developer.nvidia.com/cuda-toolkit

挑選ubuntu 16.04 runfile下載到根目錄
sudo sh cuda_8.0.44_linux.run


會出現以下安裝說明(可以按Crtl+C跳過繁雜說明)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
遇到下列問題
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 367.62?
此題答案要為n，因為我們是要裝375

下面是所以有問題及內容，請過目
Logging to /opt/temp//cuda_install_6583.log
Using more to view the EULA.
End User License Agreement
————————–

Preface
——-

The following contains specific license terms and conditions
for four separate NVIDIA products. By accepting this
agreement, you agree to comply with all the terms and
conditions applicable to the specific product(s) included
herein.

Do you accept the previously read EULA?
accept/decline/quit: accept

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 361.62?
(y)es/(n)o/(q)uit: n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
[ default is /usr/local/cuda-8.0 ]:

Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y

Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: y

Enter CUDA Samples Location
[ default is /home/textminer ]:

Installing the CUDA Toolkit in /usr/local/cuda-8.0 …
Installing the CUDA Samples in /home/textminer …
Copying samples to /home/textminer/NVIDIA_CUDA-8.0_Samples now…
Finished copying samples.

===========
= Summary =
===========

Driver: Not Selected
Toolkit: Installed in /usr/local/cuda-8.0
Samples: Installed in /home/textminer

Please make sure that
– PATH includes /usr/local/cuda-8.0/bin
– LD_LIBRARY_PATH includes /usr/local/cuda-8.0/lib64, or, add /usr/local/cuda-8.0/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run the uninstall script in /usr/local/cuda-8.0/bin

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-8.0/doc/pdf for detailed information on setting up CUDA.

***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 361.00 is required for CUDA 8.0 functionality to work.
To install the driver using this installer, run the following command, replacing with the name of this run file:
sudo .run -silent -driver

Logfile is /opt/temp//cuda_install_6583.log







<step6>
安裝完畢後，再設定一下路徑即可
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}





<last step>
最後測試：
nvidia-smi
假如沒有出現no command line，代表成功
