# From https://github.com/microsoft/AzureML-BERT/blob/master/finetune/PyTorch/dockerfile

FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN apt update && apt install git -y && rm -rf /var/lib/apt/lists/*

RUN /opt/miniconda/bin/conda update -n base -c defaults conda
RUN /opt/miniconda/bin/conda install -c r -y conda python=3.6.2 pip=20.1.1
RUN /opt/miniconda/bin/conda install -y cython=0.29.15 numpy=1.18.1

# Install cocoapi, required for drawing bounding boxes
RUN git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && python setup.py build_ext install

RUN pip install --user tensorboard cython
RUN pip install --user torch==1.5+cu101 torchvision==0.6+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install azureml-defaults
RUN pip install "azureml-dataprep[fuse]"
RUN pip install pandas pyarrow
RUN pip install opencv-python-headless
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html