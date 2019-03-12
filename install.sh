#install cuda and opencv-tegra
wget --no-check-certificate http://developer.download.nvidia.com/embedded/OpenCV/L4T_21.2/libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
wget --no-check-certificate http://developer.nvidia.com/embedded/dlc/cuda-7-toolkit-l4t-23-2
dpkg -i ~/Downloads/cuda-7-toolkit-l4t-23-2
dpkg -i libopencv4tegra-repo_l4t-r21_2.4.10.1_armhf.deb
apt-add-repository universe
apt-get update
sudo apt-get install git cuda-toolkit-7-0 libopencv4tegra libopencv4tegra-dev libopencv4tegra-python cmake3 cmake3-data python3-numpy python-numpy gcc g++ libglew-dev libtiff5-dev zlib1g-dev libjpeg-dev libpng12-dev libjasper-dev libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev libeigen3-dev libtbb-dev libgtk2.0-dev pkg-config python3-dev python3-py python3-pytest -y
echo "# add cuda bin & library path:" >> ~/.bashrc
echo "export PATH=/usr/local/cuda-7.0/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

#install opencv-python
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_extra.git
mkdir opencv/build
cd opencv/build 
#cmake -D BUILD_opencv_python3=YES -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES=../../opencv_contrib/modules -D PYTHON3_LIBRARIES=/usr/lib/arm-linux-gnueabihf/libpython3.4m.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.4/dite-packages/numpy/core/include/ ..

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_CXX_FLAGS=-Wa,-mimplicit-it=thumb \
    -DBUILD_PNG=OFF \
    -DBUILD_TIFF=OFF \
    -DBUILD_TBB=OFF \
    -DBUILD_JPEG=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_ZLIB=OFF \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_python2=ON \
    -DBUILD_opencv_python3=ON \
    -DENABLE_NEON=ON \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENMP=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_GSTREAMER=OFF \
    -DWITH_GSTREAMER_0_10=OFF \
    -DWITH_CUDA=ON \
    -DWITH_GTK=ON \
    -DWITH_VTK=OFF \
    -DWITH_TBB=ON \
    -DWITH_1394=OFF \
    -DWITH_OPENEXR=OFF \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.0 \
    -DCUDA_ARCH_BIN=3.2 \
    -DCUDA_ARCH_PTX="" \
    -DINSTALL_C_EXAMPLES=ON \
    -DINSTALL_TESTS=OFF \
    -DOPENCV_TEST_DATA_PATH=../opencv_extra/testdata \
    ../opencv
make -j4
make install
