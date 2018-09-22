

# Visual Odometry

Work in progress.

License: CC BY-NC-SA 4.0

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Install

`sudo apt-get install texlive-extra-utils gnuplot`

### Gflags

`https://github.com/gflags/gflags`

`cmake .. -DBUILD_SHARED_LIBS=ON` - Without this flag opencv can throw an error when building.

## OpenCV CMake with contrib

`wget -O opencv-3.4.2.zip https://github.com/opencv/opencv/archive/3.4.2.zip`

`wget -O opencv_contrib-3.4.2.zip https://github.com/opencv/opencv_contrib/archive/3.4.2.zip`

Unzip the archives.

```
mkdir opencv-3.4.2/build
cd opencv-3.4.2/build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.2/modules -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D WITH_OPENCL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
make -j 3
sudo make install
```

## GSTAM 

https://bitbucket.org/gtborg/gtsam


## PBA - Parallel Bundle Adjustment

http://grail.cs.washington.edu/projects/mcba/

```
cd src/sfm/pba/
make
```
