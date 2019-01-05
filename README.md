# tfsandbox

Try out and test TensorFlow programs

## install CUDA, CuDNN and TensorFlow-GPU on Ubuntu 16.04

Follow the [steps](http://blog.aicry.com/how-to-install-cuda-and-tensorflow-on-ubuntu-16-04/) and do not skip any of them.

My hardware: HP Z420 workstation installed with Nvidia GeForce GTX 1080 Ti.

In the step "Check device nodes", if the device files /dev/nvidia* do not exist, do nothing.

In the step "Install CUDA package", I used the following runfiles:
```
cuda_9.0.176.1_linux.run
cuda_9.0.176.2_linux.run
cuda_9.0.176_384.81_linux.run
cuda_9.0.176.3_linux.run
cuda_9.0.176.4_linux.run
```

In the step "Install updates for CUDA", I was asked to enter the text mode and run ```sudo service gdm stop```. I got a negative message. This was ok. To exit the text mode, you can try to press ```Ctrl+Alt+F7```. I couldn't do the next step in the text mode.

After I exited from the text mode, the dispaly resolution was changed and I couln't change it back. Don't panic. Continue...

In the step "Install graphic driver", becasue I was working behind proxy, I had to run ```sudo -E add-apt-repository ppa:graphics-drivers/ppa``` (note the option -E) to add the official Nvidia PPA to Ubuntu. I used Nviida driver version 390: ```sudo apt install nvidia-390```

In the step "Install CuDNN", I downloaded the deb files under ```cuDNN v7.3.1 (Sept 28, 2018), for CUDA 9.0```:
```
libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb
libcudnn7-dev_7.3.1.20-1+cuda9.0_amd64.deb
libcudnn7-doc_7.3.1.20-1+cuda9.0_amd64.deb
```


https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8
https://websiteforstudents.com/install-proprietary-nvidia-gpu-drivers-on-ubuntu-16-04-17-10-18-04/

### Step 1: Add The Official Nvidia PPA To Ubuntu

To work behind proxy in office, tell sudo to preserve the environment with the -E option assuming that proxies have been already configured in .bashrc file:
```
$ sudo -E add-apt-repository ppa:graphics-drivers/ppa
```
See https://askubuntu.com/questions/53146/how-do-i-get-add-apt-repository-to-work-through-a-proxy for details.

### Step 2: Update And Install Nvidia Drivers

Now that the PPA is installed, run the commands below to install the latest drivers for your system (nvidia-410 released on October 16, 2018 supports Nvidia GeForce GTX 1080 Ti that is the GPU card installed on my machine, see https://www.geforce.com/drivers/results/138959):


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
## Validate the installation of CUDNN 
You can use [the road-lane segmentation project](https://github.com/pantelis/CarND-Semantic-Segmentation-Project.git) to test if CUDNN is installed properly because the above validation tests do not invoke CUDNN.
1. Create a virtual enviroment with Python 3: ```virtualenv --python=python3 .venv_py3```
2. Run ```source .venv_py3/bin/activate``` to activate the virtual environment
3. Run ```git clone https://github.com/pantelis/CarND-Semantic-Segmentation-Project.git```
4. Download the [dataset](http://www.cvlibs.net/download.php?file=data_road.zip) used in the project: ```https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip``` and then extract the package into the directory ```CarND-Semantic-Segmentation-Project/data```
5. Dowload the pre-trained [VGG network](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) and then extract it into the same directory ```CarND-Semantic-Segmentation-Project/data```
6. Install the packages numpy, scipy and tqdm: ```pip install numpy scipy tqdm```
7. Launch the training: ```python main.py```. You will see the progress information printed on the screen. If every thing goes normally, the training will be finsihed in a few minutes.

