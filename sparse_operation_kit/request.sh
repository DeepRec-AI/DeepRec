

# judge whether it is executed by root user
user=`id -u`
if [ $user -ne 0 ]; then
echo "[ERROR]: You don't have enough privilege to install SOK."
exit
fi

#install nccl
#For downloading NCCL you have to registered here https://developer.nvidia.com/developer-program/signup
wget https://developer.nvidia.com/compute/machine-learning/nccl/secure/2.11.4/ubuntu1804/x86_64/nccl-local-repo-ubuntu1804-2.11.4-cuda11.0_1.0-1_amd64.deb
dpkg -i nccl-local-repo-ubuntu1804-2.11.4-cuda11.0_1.0-1_amd64.deb
apt-get update
apt install libnccl2=2.11.4-1+cuda11.0 libnccl-dev=2.11.4-1+cuda11.0
cp /usr/include/nccl.h /usr/local/cuda/include/nccl.h

#install cmake
apt-get install cmake

#install openmpi
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
tar zxf openmpi-4.1.1.tar.gz
cd openmpi-4.1.1 
./configure --prefix=/usr/local/openmpi
make -j && make -j install
echo 'MPI_HOME=/usr/local/openmpi' >> ~/.bashrc
echo 'export PATH=${MPI_HOME}/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export MANPATH=${MPI_HOME}/share/man:$MANPATH' >> ~/.bashrc
source ~/.bashrc
mpirun --version
rm openmpi-4.1.1.tar.gz && rm -rf openmpi-4.1.1

#install horovod
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod
horovodrun --check-build


echo "Successfully installed."

