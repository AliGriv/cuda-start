# cuda-start
A project to start a simple starting project with CUDA

## CUDA Installation
My operating system is Ubuntu 16.04 and the Kernel Release is 4.15.0-120-generic. The command is:
```
uname -a
```
The steps for installation was taken from [DIX SOFT tutorial](https://youtu.be/8Xy1Uqq9Hbk) and I installed CUDA-8.0.
I'm using CLion and my next problem was similar to [this thread](https://youtrack.jetbrains.com/issue/CPP-19502).
One solution was to add /usr/local/cuda-<version>/bin to PATH in the /etc/environment configuration file.
After googling this issue, a similar dicussion was found [here](https://askubuntu.com/questions/866161/setting-path-variable-in-etc-environment-vs-profile). The problem is how to change and add the path for all users in ```/etc/environment```.
The steps that I went through was:
  1. Copy the current file for backup.
  ``` sudo cp /etc/environment /etc/environment.orig ```
  2. Edit the existing file
  ```sudo nano -w /etc/environment```
  Previously, the content was
```PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"$```.
  It was changed to:
```PATH="/usr/local/cuda-8.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games"```.
  
  
Another error that was encountered was about not being able to find the library directory. Similar to previous step, one solution would be adding the directory for CUDA   in a way that all the users and environments have access, i.e. adding it the shared library of the system. [This article](https://blog.andrewbeacock.com/2007/10/how-to-add-shared-libraries-to-linuxs.html) was useful. The steps were as following:
1. Create a new file in /etc/ld.so.conf.d/ called .conf
```sudo touch libcuda.conf```
2. Edit the file and add a line per directory of shared libraries (*.so files)
  ```sudo gedit libcuda.conf```
3. The following was added to the blank file and saved:
```
#cuda library
/usr/local/cuda-8.0/lib64
```
4. Reload the list of system-wide library paths:
```
sudo ldconfig
```


## Results

Up to now, it seems that cudaMalloc takes a lot of time. It is to allocate memory beforehand so that we do not include it in measuring the performance. After cudaMalloc, cudaMemcpy also takes some time which the unnecessary part can be avoided. Other than that, the program seems to be working after all!!
