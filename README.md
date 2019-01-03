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
Now that the PPA is installed, run the commands below to install the latest drivers for your system (nvidia-410 released on October 16, 2018 supports Nvidia GeForce GTX 1080 Ti that is the GPU card installed on my machine):
```
$ sudo apt update
$ sudo apt install nvidia-410
```
After installing the drivers above reboot your system for the new drivers to be enabled on the systems.

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


