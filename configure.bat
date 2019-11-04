set ARCH=sm_35
if NOT [%1]==[] set ARCH=%1

mkdir cmake-build 2> nul
cd cmake-build
cmake ..\\ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-arch=%ARCH% --use_fast_math -Xptxas -O3" -G "Visual Studio 15 2017 Win64"
cd ..
