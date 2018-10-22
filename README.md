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



