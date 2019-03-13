#install cuda and opencv-tegra
wget --no-check-certificate http://developer.download.nvidia.com/embedded/OpenCV/L4T_21.2/libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
wget --no-check-certificate http://developer.nvidia.com/embedded/dlc/cuda-7-toolkit-l4t-23-2
dpkg -i ~/Downloads/cuda-7-toolkit-l4t-23-2
dpkg -i libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
apt-add-repository universe
apt-get update
sudo apt-get install git \
	cuda-toolkit-7-0 \
	libopencv4tegra \
	libopencv4tegra-dev \
	libopencv4tegra-python \
	cmake3 \
	cmake3-data \
	python-numpy \
	gcc \
	g++ \
	libglew-dev \
	libtiff5-dev \
	zlib1g-dev \
	libjpeg-dev \
	libpng12-dev \
	libjasper-dev \
	libavcodec-dev \
	libavformat-dev \
	libavutil-dev \
	libpostproc-dev \
	libswscale-dev \
	libeigen3-dev \
	libtbb-dev \
	libgtk2.0-dev \
	pkg-config \
	python3-dev \
	python3-numpy \
	python3-py \
	python3-pytest \
	python3-pip \
	-y
echo "# add cuda bin & library path:" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-7.0/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

#install opencv-python
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_extra.git
git clone https://github.com/opencv/opencv_contrib.git
mkdir opencv/build
cd opencv/build 

#remove CMakeCache.txt when error denote about "source dirictory do not exist"
cmake -D BUILD_opencv_python3=YES -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES=../../opencv_contrib/modules -D PYTHON3_LIBRARIES=/usr/lib/arm-linux-gnueabihf/libpython3.4m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.4/dite-packages/numpy/core/include/ ..

make -j4
make install

pip3 insatll pyserial
