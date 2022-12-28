# Installing dlib using `conda` with CUDA enabled

Prerequisite: `conda` and/or `miniconda` are already installed

1. Create a conda environment.
```console
$ conda create -n dlib-gpu python=3.8 cmake ipython
```

2. Activate the environment.
```console
$ conda activate dlib-gpu
```

3. Install CUDA and cuDNN using nvidia references (Adding the required libraries to the PATH)

https://www.youtube.com/watch?v=OEFKlRSd8Ic&t=1030s

This video shows how to enable CUDA and CUDNN to the path

Extra information related to the video, there is no need to create the "TOOLS" folder, if you copied the cudnn into the CUDA file, windows will be able to find it



4. Install dlib.
Clone and build dlib from source and check here in case of future modifications

http://dlib.net/compile.html


```console
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build
$ cd build
$ cmake -G "Visual Studio 15 2017" -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 ..
$ cmake --build . --config Release
$ cd ..

# In the next step you should select the visual studio that suits you better, I tried this with the 2022 and didn't compile, but 2019 and 2017 worked well,
also, seems like there is a problem with the GIF_SUPPORT library which will collapse the installation, so specify that is not needed

$ python setup.py install -G "Visual Studio 15 2017" --set USE_AVX_INSTRUCTIONS=1 --set DLIB_USE_CUDA=1 --no DLIB_GIF_SUPPORT
```

5. Test dlib
```console
(dlib) $ ipython
Python 3.8.12 (default, Oct 12 2021, 13:49:34)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.27.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import dlib

In [2]: dlib.DLIB_USE_CUDA
Out[2]: True

In [3]: print(dlib.cuda.get_num_devices())
1
```
