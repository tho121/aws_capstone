# SageMaker PyTorch image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04

RUN pip3 install sagemaker-training
RUN pip3 install datasets -U

#ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
#ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
#COPY train_vit_hf.py /opt/ml/code/train_vit_hf.py

# Defines cifar10.py as script entrypoint 
#ENV SAGEMAKER_PROGRAM train_vit_hf.py

