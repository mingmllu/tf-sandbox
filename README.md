# tfsandbox

Try out and test TensorFlow programs

## Install CUDA, CuDNN and TensorFlow-GPU on Ubuntu 16.04

Follow the [steps](http://blog.aicry.com/how-to-install-cuda-and-tensorflow-on-ubuntu-16-04/) and do not skip any step.

Hardware: HP Z420 workstation installed with Nvidia GeForce GTX 1080 Ti.

In the step "Check device nodes", if the device files /dev/nvidia* do not exist, do nothing.

In the step "Install CUDA package", the following runfiles can be used:
```
cuda_9.0.176.1_linux.run
cuda_9.0.176.2_linux.run
cuda_9.0.176_384.81_linux.run
cuda_9.0.176.3_linux.run
cuda_9.0.176.4_linux.run
```

In the step "Install updates for CUDA", after entering the text mode, you are asked to run ```sudo service gdm stop```. You may get a negative message. This is okay. To exit the text mode, you can try to press ```Ctrl+Alt+F7```. In the text mode, you can't do the next step.

After you exit the text mode, the dispaly resolution may be changed! Don't panic. Continue...

In the step "Install graphic driver", if you are working [behind proxy](https://askubuntu.com/questions/53146/how-do-i-get-add-apt-repository-to-work-through-a-proxy), you have to use the option -E to tell sudo to preserve the environment assuming that proxies have been already configured in .bashrc file. So run ```sudo -E add-apt-repository ppa:graphics-drivers/ppa``` to add the official Nvidia PPA to Ubuntu. You may use Nvidia driver version 390: ```sudo apt install nvidia-390```

In the step "Install CuDNN", you can download the deb files under ```cuDNN v7.3.1 (Sept 28, 2018), for CUDA 9.0```:
```
libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb
libcudnn7-dev_7.3.1.20-1+cuda9.0_amd64.deb
libcudnn7-doc_7.3.1.20-1+cuda9.0_amd64.deb
```
[Ref1](https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8)
[Ref2](https://websiteforstudents.com/install-proprietary-nvidia-gpu-drivers-on-ubuntu-16-04-17-10-18-04/)

## Install Anaconda on Ubuntu 16.04 LTS
1. Download the latest Anaconda installer bash script at https://www.anaconda.com/download/#linux. Don't worry about Anaconda's Python version. If you need a specific Python version, you can create a conda environment with the Python version. Once it is finished, you should see the file "Anaconda3-2018.12-Linux-x86_64.sh" in ~/Downloads.
2. Change directory to ~/Downloads, run ```bash Anaconda3-2018.12-Linux-x86_64.sh``` 
3. [Create a virtual environment](https://conda.io/docs/user-guide/tasks/manage-python.html). Note that if you do not use the metapackage ```anaconda``` that includes all of the Python packages comprising the Anaconda distribution, you may have problem with running tensorflow-gpu.
```
conda create -n yourenvname python=x.x anaconda
```
4. Remove a conda environment
```
$ conda env remove --name YOUR_ENV_NAME
```
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
## Validate the installation of CuDNN - Approach 1
You can use [the road-lane segmentation project](https://github.com/pantelis/CarND-Semantic-Segmentation-Project.git) to test if CUDNN is installed properly because the above validation tests do not invoke CuDNN.
1. Create a virtual enviroment with Python 3: ```virtualenv --python=python3 .venv_py3```
2. Run ```source .venv_py3/bin/activate``` to activate the virtual environment
3. Run ```git clone https://github.com/pantelis/CarND-Semantic-Segmentation-Project.git```
4. Download the [dataset](http://www.cvlibs.net/download.php?file=data_road.zip) used in the project: ```https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip``` and then extract the package into the directory ```CarND-Semantic-Segmentation-Project/data```
5. Dowload the pre-trained [VGG network](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip) and then extract it into the same directory ```CarND-Semantic-Segmentation-Project/data```
6. Install the packages numpy, scipy and tqdm: ```pip install numpy scipy tqdm```
7. Launch the training: ```python main.py```. You will see the progress information printed on the screen. If every thing goes normally, the training will be finsihed in a few minutes.

## Validate the installation of CuDNN - Approach 2
[Benchmark your GPU-capable platform](https://www.tensorflow.org/guide/performance/benchmarks)

[Methodology](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)
1. Create a virtual environment
```
$ virtualenv --python=python3 _venv_gpu_py3_benchmark
```
2. Enter the virtual environment:
```
$ source _venv_gpu_py3_benchmark/bin/activate
```
3. Clone the TensorFlow Benchmarks repo
```
$ git clone https://github.com/tensorflow/benchmarks.git
```
4. Install TensorFlow-GPU
```
$ pip install tensorflow-gpu
```
5. Check the version of the installed TensorFlow
6. Check out the git repo branch corresponding to the installed TensorFlow. For example, if it is TensorFlow 1.12, do as below
```
$ git branch cnn_tf_v1.12_compatible
```
7. Change dircetory ```cd benchmarks/scripts/tf_cnn_benchmarks``` and run the test on single-GPU machine:
```
$ python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=parameter_server
2019-01-08 23:12:36.719973: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-01-08 23:12:36.720494: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.582
pciBusID: 0000:05:00.0
totalMemory: 10.91GiB freeMemory: 10.12GiB
2019-01-08 23:12:36.720515: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-08 23:12:36.979361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-08 23:12:36.979404: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-01-08 23:12:36.979412: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-01-08 23:12:36.979681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9779 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:05:00.0, compute capability: 6.1)
TensorFlow:  1.12
Model:       vgg16
Dataset:     imagenet (synthetic)
Mode:        BenchmarkMode.TRAIN
SingleSess:  False
Batch size:  32 global
             32.0 per device
Num batches: 100
Num epochs:  0.00
Devices:     ['/gpu:0']
Data format: NCHW
Optimizer:   sgd
Variables:   parameter_server
==========
Generating training model
Initializing graph
W0108 23:12:37.619437 140034087950080 tf_logging.py:125] From /home/mmlu/Playground/benchmarks/scripts/tf_cnn_benchmarks/benchmark_cnn.py:2157: Supervisor.__init__ (from tensorflow.python.training.supervisor) is deprecated and will be removed in a future version.
Instructions for updating:
Please switch to tf.train.MonitoredTrainingSession
2019-01-08 23:12:37.729993: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-08 23:12:37.730060: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-08 23:12:37.730069: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-01-08 23:12:37.730077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-01-08 23:12:37.730320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 9779 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:05:00.0, compute capability: 6.1)
I0108 23:12:37.927248 140034087950080 tf_logging.py:115] Running local_init_op.
I0108 23:12:37.956495 140034087950080 tf_logging.py:115] Done running local_init_op.
Running warm up
Done warm up
Step	Img/sec	total_loss
1	images/sec: 128.3 +/- 0.0 (jitter = 0.0)	7.268
10	images/sec: 131.8 +/- 0.5 (jitter = 2.0)	7.238
20	images/sec: 132.2 +/- 0.3 (jitter = 1.3)	7.262
30	images/sec: 132.3 +/- 0.2 (jitter = 1.4)	7.228
40	images/sec: 132.2 +/- 0.3 (jitter = 1.4)	7.295
50	images/sec: 132.1 +/- 0.2 (jitter = 1.3)	7.284
60	images/sec: 132.2 +/- 0.2 (jitter = 1.3)	7.253
70	images/sec: 132.2 +/- 0.2 (jitter = 1.3)	7.258
80	images/sec: 132.2 +/- 0.2 (jitter = 1.5)	7.251
90	images/sec: 132.2 +/- 0.2 (jitter = 1.4)	7.257
100	images/sec: 132.2 +/- 0.2 (jitter = 1.4)	7.260
----------------------------------------------------------------
total images/sec: 132.11
----------------------------------------------------------------
```

## Start a GPU container with Python 3, using the Python interpreter
```
$ docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu-py3 python
```

## [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
```
$ pip install tensorflow-gpu

$ sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
$ pip install --user Cython
$ pip install --user contextlib2
$ pip install --user jupyter
$ pip install --user matplotlib
$ cd ~/Downloads
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI
$ make
$ cp -r pycocotools <path_to_tensorflow>/models/research/

# From tensorflow/models/research/
$ wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
$ unzip protobuf.zip

# From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.

# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# This command needs to run from every new terminal you start. If you wish to avoid running 
# this manually, you can add it as a new line to the end of your ~/.bashrc file, replacing #
# `pwd` with the absolute path of tensorflow/models/research on your system.

# Make sure you are in the virtual environment, from tensorflow/models/research/, testing the installation
$ python object_detection/builders/model_builder_test.py
......................
----------------------------------------------------------------------
Ran 22 tests in 0.104s
```
[Slow inference speed of object detection models and a hack as solution](https://github.com/tensorflow/models/issues/3270)
