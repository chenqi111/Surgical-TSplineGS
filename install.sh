# # install surgical_tsplinegs environment
# ENV_NAME=surgical_tsplinegs

# conda remove -n $ENV_NAME --all -y
# conda create -n $ENV_NAME python=3.7 
# conda activate $ENV_NAME
# export CUDA_HOME=$CONDA_PREFIX

# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# conda install nvidia/label/cuda-11.7.0::cuda
# conda install nvidia/label/cuda-11.7.0::cuda-nvcc
# conda install nvidia/label/cuda-11.7.0::cuda-runtime
# conda install nvidia/label/cuda-11.7.0::cuda-cudart
# conda install -c conda-forge ld_impl_linux-64

# pip install -e submodules/simple-knn
# pip install -e submodules/co-tracker
# pip install -r requirements.txt

# install unidepth environment
UNIDEPTH_ENV_NAME=unidepth_surgical_tsplinegs
conda remove -n $UNIDEPTH_ENV_NAME --all -y
conda create -n $UNIDEPTH_ENV_NAME python=3.10
conda activate $UNIDEPTH_ENV_NAME

pip install -r requirements_unidepth.txt
conda install -c conda-forge ld_impl_linux-64
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda install nvidia/label/cuda-12.1.0::cuda
conda install nvidia/label/cuda-12.1.0::cuda-nvcc
conda install nvidia/label/cuda-12.1.0::cuda-runtime
conda install nvidia/label/cuda-12.1.0::cuda-cudart
conda install nvidia/label/cuda-12.1.0::libcusparse
conda install nvidia/label/cuda-12.1.0::libcublas
cd submodules/UniDepth/unidepth/ops/knn;bash compile.sh;cd ../../../../../
pip install -e submodules/UniDepth

# install depthanything
mkdir -p submodules/mega-sam/Depth-Anything/checkpoints