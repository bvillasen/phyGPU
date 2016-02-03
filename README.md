# phyGPU

A python environment that uses pyCUDA to simulate and visualize physics problems.

##Installation
For using phyGPU you need CUDA and a set of simple prerequisites as well as virtualenv.

###Installing CUDA
For a detailed installation guide refer to the Installation Guide for Linux.
For a quick installation follow the next instructions:
- Get the most recent version of the CUDA Toolkit .deb file [HERE.](https://developer.nvidia.com/cuda-downloads)

- Install the repository meta-data, update the apt-get cache, and install CUDA:
```
$ sudo dpkg --install cuda-repo-<distro>-<version>.<architecture>.deb
$ sudo apt-get update
$ sudo apt-get install cuda
```

- Reboot the system to load the NVIDIA drivers.
- Set up the development environment by modifying the PATH and
LD_LIBRARY_PATH variables: by adding the next lines at the end of your .bashrc file
```
$ export PATH=/usr/local/cuda-7.5/bin:$PATH
$ export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
```

###Installing prerequisites
```
$ sudo apt-get install python-dev python-pip libpng-dev libfreetype6-dev
```

###Installing Virtualenv
A Virtual Environment is a tool to keep the dependencies required by different projects
in separate places, by creating virtual Python environments for them.
The easiest way to install virtualenv is by running:
```
$ sudo pip install virtualenv
```
