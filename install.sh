#!/bin/bash
CURRENT_DIR=$(pwd)
HOME_DIR=~
VIRT_ENV_DIR="/home/bruno/apps/virtualenv-15.1.0"
#The first argument is the directory were the virtualenv will be created
ENV_DIR="$1"
ENV_NAME="phyGPU_env"
BIN_DIR=$ENV_DIR/$ENV_NAME/binaries

#Create virtual enviroment
cd $ENV_DIR
python $VIRT_ENV_DIR/virtualenv.py $ENV_NAME
chmod 777 $ENV_NAME/bin/activate
#
# # Create activation comand under binaries directory
# mkdir $BIN_DIR
# touch $BIN_DIR/activate_phyGPU
# LINE="echo Welcome to phyGPU to exit type deactivate"
# echo $LINE >> $BIN_DIR/activate_phyGPU
# LINE="source $ENV_DIR/$ENV_NAME/bin/activate"
# echo $LINE >> $BIN_DIR/activate_phyGPU
# chmod 777 $BIN_DIR/activate_phyGPU
#
# #Add binaries directory to your path
# LINE="export PATH=$BIN_DIR:\$PATH"
# LAST_LINE=$(awk '/./{line=$0} END{print line}' ~/.bashrc)
# if [ "$LAST_LINE" != "$LINE" ]; then
#   echo "Adding binaries directory to .bashrc file"
#   echo $LINE >> ~/.bashrc
# fi
#
# # Activate the pycuda enviroment and install packages
# . ./$ENV_NAME/bin/activate
# pip install ipython[all]
# pip install numpy
# pip install matplotlib
# pip install PyOpenGL
#
# #########################################################
# #INSTALL PYCUDA
# wget https://pypi.python.org/packages/source/p/pycuda/pycuda-2015.1.3.tar.gz
# tar xzvf pycuda-2015.1.3.tar.gz -C $ENV_DIR/$ENV_NAME/
# rm pycuda-2015.1.3.tar.gz
# cd $ENV_DIR/$ENV_NAME/pycuda-2015.1.3
# ./configure.py
# #Replace line in siteconf.py to enable graphics
# #Create temporary file with new line in place
# cat siteconf.py | sed -e "s/CUDA_ENABLE_GL = False/CUDA_ENABLE_GL = True/" > siteconf_temp.py
# #Copy the new file over the original file
# mv siteconf_temp.py siteconf.py
# make -j 4
# python setup.py install
# pip install .
# cd $ENV_DIR/$ENV_NAME
# #########################################################
#
# #Download projects to projects directory
# mkdir projects
# cd projects
# git clone https://github.com/bvillasen/tools.git
# git clone https://github.com/bvillasen/volumeRender.git
# git clone https://github.com/bvillasen/animation2D.git
# git clone https://github.com/bvillasen/isingModel.git
