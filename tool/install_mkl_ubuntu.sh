wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update
sudo apt-cache search mkl
sudo apt-get install -qq intel-mkl-core-rt-2018.2-199

source /opt/intel/compilers_and_libraries_2018.2.199/linux/bin/compilervars.sh intel64
