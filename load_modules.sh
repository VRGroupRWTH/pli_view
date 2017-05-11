echo OpenGL Version Info:
glxinfo | grep "version"

echo Purging modules.
module --force purge

echo Setting modules directories.
module use /usr/local/software/jureca/Stages/2016a/modules/all
module use /usr/local/software/jureca/Stages/2017a/modules/all
module use /usr/local/software/jureca/Stages/Devel-2017a/modules/all

echo Loading tools.
module load GCCcore/.5.4.0				# Dependency of Cmake.
module load CMake/3.7.2 				# Tool of PLIVIS.
module load GCC/5.4.0 					# Tool of PLIVIS.

echo Loading dependencies.
module load MVAPICH2/2.2-GDR 			# Dependency of Boost.
module load Boost/1.63.0-Python-2.7.13  # Dependency of PLIVIS. Hopefully includes original C++ version.
module load CUDA/8.0.61 				# Dependency of PLIVIS.
module load HDF5/1.8.18-serial 			# Dependency of PLIVIS.
module load Compiler/GCCcore/5.4.0/Qt5/ # Dependency of PLIVIS.
