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
###Installing phyGPU
To install phyGPU make sure you are located in the phyGPU directory and just run the setup file (do not use sudo):
```
$ python setup.py
```
You will be asked the location where the phyGPU environment should be installed, by hitting ENTER the default phyGPU directory will be used. If you want to use another directory type the ABSOLUTE path and hit ENTER, then just accept the confirmation.


This will create a directory called phyGPU_env and install a set of commonly used python modules: iPython(notebook), numpy, matplotlib and pyCUDA and pyOpenGL for real-time simulation and visualization of physics problems.  

##Activating phyGPU
To use phyGPU you need to activate it's phython environment, this will change the python PATH to the phyGPU python interpreter, so instead of using the "global" version of python you will be using the phyGPU "private" python version. To activate phyGPU run:
```
$ . activate_phyGPU
```
NOTE the dot at the beginning of the command, it will make the command to take effect in the current shell, without the dot the command will take effect in a child shell and phyGPU won't be active. Also you could use the alternative form:
```
$ source activate_phyGPU
```
