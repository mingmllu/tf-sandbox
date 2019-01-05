# tfsandbox

Try out and test TensorFlow programs

## Install NVIDIA Graphics Driver on Ubuntu 16.04 LTS

https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8

http://blog.aicry.com/how-to-install-cuda-and-tensorflow-on-ubuntu-16-04/

https://websiteforstudents.com/install-proprietary-nvidia-gpu-drivers-on-ubuntu-16-04-17-10-18-04/

### Step 1: Add The Official Nvidia PPA To Ubuntu

To work behind proxy in office, tell sudo to preserve the environment with the -E option assuming that proxies have been already configured in .bashrc file:
```
$ sudo -E add-apt-repository ppa:graphics-drivers/ppa
```
See https://askubuntu.com/questions/53146/how-do-i-get-add-apt-repository-to-work-through-a-proxy for details.

### Step 2: Update And Install Nvidia Drivers

Now that the PPA is installed, run the commands below to install the latest drivers for your system (nvidia-410 released on October 16, 2018 supports Nvidia GeForce GTX 1080 Ti that is the GPU card installed on my machine, see https://www.geforce.com/drivers/results/138959):
```
$ sudo apt update
$ sudo apt install nvidia-410
```
After installing the drivers above reboot your system for the new drivers to be enabled on the systems. Then use the lsmod command to check your installation status with the following command. It will list all currently loaded kernel modules in Linux, then filter only nvidia using grep command.
```
$ lsmod | grep nvidia
```
You should see the following installation status
```
nvidia_uvm            790528  0
nvidia_drm             40960  4
nvidia_modeset       1040384  8 nvidia_drm
nvidia              16560128  334 nvidia_modeset,nvidia_uvm
drm_kms_helper        172032  1 nvidia_drm
drm                   401408  7 nvidia_drm,drm_kms_helper
ipmi_msghandler        53248  3 nvidia,ipmi_devintf,ipmi_si
```
Check device node:
```
$ ls -l /dev/nvidia*
```
You should see something like this
```
crw-rw-rw- 1 root root 195,   0 Oct 22 12:40 /dev/nvidia0
crw-rw-rw- 1 root root 195, 255 Oct 22 12:40 /dev/nvidiactl
crw-rw-rw- 1 root root 195, 254 Oct 22 12:40 /dev/nvidia-modeset
crw-rw-rw- 1 root root 236,   0 Oct 22 12:40 /dev/nvidia-uvm
```
Some times updates do not work well as expected. If you face any issues with the latest drivers installation such as black screen on startup, you can remove it as follows.
```
$ sudo apt-get purge nvidia*
```
If you want to completely remove graphics-drivers PPA as well, run the following command to remove PPA.
```
$ sudo apt-add-repository --remove ppa:graphics-drivers/ppa
```

## Install CUDA 10.0

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal

Download Installer for Linux Ubuntu 16.04 x86_64.

CD to the Downloads directory, run the following:
```
$ chmod +x cuda_10.0.130_410.48_linux.run
$ ./cuda_10.0.130_410.48_linux.run --extract=$HOME
```
You'll see the three runfiles:
```
-rwxrwxr-x 1 mmlu mmlu 1834769129 Oct 22 23:44 cuda-linux.10.0.130-24817639.run*
-rwxrwxr-x 1 mmlu mmlu   86469433 Oct 22 23:44 cuda-samples.10.0.130-24817639-linux.run*
-rwxrwxr-x 1 mmlu mmlu  105930536 Oct 22 23:44 NVIDIA-Linux-x86_64-410.48.run*
```
Run
```
$ sudo ./cuda-linux.10.0.130-24817639.run
$ sudo ./cuda-samples.10.0.130-24817639-linux.run
```
After the installation finishes, configure the runtime library, run
```
$ sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
$ sudo ldconfig
```

Run 

```
$ sudo vi /etc/profile.d/cuda.sh
```
and add the following exports to /etc/profile.d/cuda.sh:

```
export PATH=/usr/local/cuda/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Install CUDA Profile Tools Interface:

```
$ sudo apt-get install -y libcupti-dev
```

And add the following line to /etc/profile.d/cuda.sh:

```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
```

At the end, reboot your system:

```
$ sudo reboot
```

Verify the installation

```
$ cd /usr/local/cuda/samples
$ sudo make
```

## Install Anaconda on Ubuntu 16.04 LTS
1. Download the latest Anaconda installer bash script at https://www.anaconda.com/download/#linux. Once it is finished, you should see the file "Anaconda3-2018.12-Linux-x86_64.sh" in ~/Downloads.
2. Change directory to ~/Downloads, run ```bash Anaconda3-2018.12-Linux-x86_64.sh``` 

## Install TensorFlow in Anaconda

1. Install TensorFlow CPU: ```conda install tensorflow``` ([Stop Installing Tensorflow using pip for performance sake](https://towardsdatascience.com/stop-installing-tensorflow-using-pip-for-performance-sake-5854f9d9eb0c))
2. Install TensorFlow GPU: ```conda install -c anaconda tensorflow-gpu```. Unfortunately, tf.Session does not work in my workstation. So I use pip to install TensorFlow GPU.
3. Validate the installation of TensorFlow
```
import tensorflow as tf   
from tensorflow.python.client import device_lib
device_lib.list_local_devices() #this will show all CPU's and GPU's on your system
```
```
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```
```
import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
```
## Train a neural network for semantic segmentation to validate the installation of CUDNN 
You can use [the road-lane segmentation project](https://github.com/pantelis/CarND-Semantic-Segmentation-Project.git) to test if CUDNN is installed properly because the above validation tests do not invoke CUDNN.
1. Create a virtual enviroment with Python 3: ```virtualenv --python=python3 .venv_py3```
2. Run ```source .venv_py3/bin/activate``` to activate the virtual environment
3. Run ```git clone https://github.com/pantelis/CarND-Semantic-Segmentation-Project.git```
4. Download the [dataset](http://www.cvlibs.net/download.php?file=data_road.zip) used in the project: ```https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip``` and then extract the package into the directory ```CarND-Semantic-Segmentation-Project/data```
5. Dowload the pre-trained [VGG network](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) and then extract it into the same directory ```CarND-Semantic-Segmentation-Project/data```
6. Install the packages numpy, scipy and tqdm: ```pip install numpy scipy tqdm```
7. Launch the training: ```python main.py```. You will see the progress information printed on the screen. If every thing goes normally, the training will be finsihed in a few minutes.

