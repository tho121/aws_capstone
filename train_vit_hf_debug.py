import datasets
from transformers import ViTForImageClassification, Trainer, TrainingArguments,default_data_collator,ViTFeatureExtractor
from datasets import load_dataset, load_metric, disable_caching
import torch
import logging
import sys
import argparse
import os
import numpy as np
import subprocess
from transformers.trainer_utils import get_last_checkpoint
from torchvision.transforms import RandomHorizontalFlip, Normalize, Compose, ToTensor

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

if __name__ == "__main__":

    print("datasets version: ")
    print(datasets.__version__)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str,default="/opt/ml/model")
    parser.add_argument("--extra_model_name", type=str,default="sagemaker")
    parser.add_argument("--dataset", type=str,default="cifar10")
    parser.add_argument("--task", type=str,default="image-classification")
    parser.add_argument("--use_auth_token", type=str, default="")

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=str, default=2e-5)

    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_name)
    flip = RandomHorizontalFlip(p=0.5)

    def train_img_transforms(example_batch):
        images = [flip(image.convert("RGB")) for image in example_batch["pixel_values"]]
        inputs = feature_extractor(images, return_tensors='pt')

        inputs['labels'] = [x for x in example_batch['label']]

        return inputs

    def test_img_transforms(example_batch):
        images = [image.convert("RGB") for image in example_batch["pixel_values"]]
        inputs = feature_extractor(images, return_tensors='pt')

        inputs['labels'] = [x for x in example_batch['label']]

        return inputs


    # load datasets
    train_dataset = load_dataset('imagefolder', data_dir=args.training_dir, split='train')
    test_dataset = load_dataset('imagefolder', data_dir=args.test_dir, split='train')
    
    train_dataset = train_dataset.rename_column("image", "pixel_values")
    test_dataset = test_dataset.rename_column("image", "pixel_values")

    num_classes = train_dataset.features['label'].num_classes
    labels = train_dataset.features['label'].names

    #feature extractor transformations need to be applied after image transformations
    #includes normalization and conversion to tensor
    #but way too big to save transformations, must be applied each time image is sampled
    train_dataset.set_transform(train_img_transforms)
    test_dataset.set_transform(test_img_transforms)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    metric_name = "accuracy"
    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)


    print("Current GPU:")
    print_gpu_utilization()

    model = ViTForImageClassification.from_pretrained(args.model_name, 
                num_labels=num_classes,
                id2label={str(i): c for i, c in enumerate(labels)},
                label2id={c: str(i) for i, c in enumerate(labels)})

    model.to(device)

    print("GPU after model instanced:")
    print_gpu_utilization()
    
    print(train_dataset[0])
    
    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        dataloader_num_workers=2,
    )
    
    
    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        tokenizer=feature_extractor
    )

    # train model
    if get_last_checkpoint(args.output_dir) is not None:
        logger.info("***** continue training *****")
        last_checkpoint = get_last_checkpoint(args.output_dir)
        result = trainer.train(resume_from_checkpoint=last_checkpoint)
        print_summary(result)
    else:
        result = trainer.train()
        print_summary(result)

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(args.output_dir)

    if args.use_auth_token != "":
        kwargs = {
            "finetuned_from": args.model_name.split("/")[1],
            "tags": "image-classification",
            "dataset": args.dataset,
        }
        repo_name = (
            f"{args.model_name.split('/')[1]}-{args.task}"
            if args.extra_model_name == ""
            else f"{args.model_name.split('/')[1]}-{args.task}-{args.extra_model_name}"
        )
 
        trainer.push_to_hub(
            repo_name=repo_name,
            use_auth_token=args.use_auth_token,
            **kwargs,
        )
