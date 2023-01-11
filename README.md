# AWS Warehouse Classifier

For my Udacity Machine Learning Capstone project, I selected the Inventory Monitoring at Distributions Centers project. In distribution centers, there are robots that hold bins containing a varying number of items. However, there may be errors resulting in the expected number of items in the bin not matching the actual number. Detecting these errors early can result in substantial benefits when applied to a large-scale operation, such as in an Amazon Distribution Center. The actual number of items can be identified by evaluating an image of the bin, taken from a top-down view. The goal of this project is to train a deep learning model that can classify an image of the bin’s contents as a bin with the correct number of items.

## Project Set Up and Installation

This project uses AWS Sagemaker to creating training jobs that run on AWS resources. However, a Jupyter Notebook can be used to run a local instance for testing and debugging purposes. The `sagemaker_vit_local.ipynb` file is setup to run locally. It relies on a docker image that can be built using the Dockerfile found in the `docker_debugging` folder.

    cd docker_debugging

    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

    docker build -t vit-debug-container .

These lines access the pre-built images hosted by AWS, which are needed since the Dockerfile extends a specific image, and then builds the Dockerfile for local debugging.


## Dataset

The AWS Warehouse dataset features 535,234 image and json file pairs. Each image has a top-down view of a bin that will contain 0-5 items. For this project, only images with 1-5 items are used. The images are resized to 224x244. The train dataset size is 198,000 and the test dataset size is 22,000. 

https://github.com/awslabs/open-data-docs/tree/main/docs/aft-vbi-pds

## Model Training

Following the implementation outlined in the paper Amazon Inventory Reconciliation using AI by Pablo Rodriguez Bertorello, Sravan Sripada, and Nutchapol Dendumrongsup (https://github.com/pablo-tech/Image-Inventory-Reconciliation-with-SVM-and-CNN/blob/master/ProjectReport.pdf), this project replicates the training for a ResNet34 model to achieve similar test accuracy as a benchmark. The ResNet34 model uses the implementation from the Torchvision package with pretrained weights, and uses the following hyperparameters for training.

    hyperparameters = {
        'batch_size': 128, 
        'lr': 0.001, 
        'epochs': 10, 
    }


The same dataset is used to train a ViT model to evaluate and compare the performance of transformer models in the computer vision domain. The ViT model uses the implementation from the Huggingface package, also with pretrained weights, and uses the following hyperparameters for training.

    hyperparameters = {
        'num_train_epochs': 5, # train epochs
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 64, # batch size
        'per_device_eval_batch_size': 64, # batch size
        'dataloader_workers': 4,
        'model_name': "google/vit-base-patch16-224-in21k", # model which will be trained on
    }



**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model.

## Machine Learning Pipeline
The `dataset.ipynb` notebook parses the metadata from the AWS Warehouse dataset to identify the target images to download. These images are then downloaded locally and resized. They're then uploaded to the s3 bucket.

The `sagemaker.ipynb` notebook starts the training job for the ResNet34 model. The `sagemaker_vit.ipynb` notebook starts training job for the ViT model. The `sagemaker_vit_local.ipynb` notebook is intended only for local debugging.

Cloudwatch provides logs which can be filtered for the relevant metrics. They can be copied from Cloudwatch and saved a csv files. These files are expected to reside in the `metrics` folder. The `metrics.ipynd` notebook generates the appropriate graphs.