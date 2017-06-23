echo OpenGL Version Info:
glxinfo | grep "version"

echo Purging modules.
module --force purge

echo Setting modules directories.
module use Stages/2017a

echo Loading dependencies.
module load GCC/5.4.0
module load MVAPICH2/2.2-GDR
module load CUDA # has dependency nvidia/.driver
module load CMake
module load Boost
module load HDF5
module load Qt5/.5.8.0