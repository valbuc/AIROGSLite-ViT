print("ENtered Script")

import os
from PIL import Image
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification, AdamW
import pandas as pd
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision.transforms import (Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    GaussianBlur,
                                    RandomRotation, #  rotation
                                    Resize, 
                                    ToTensor)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping


# Params
train_batch_size = 32
eval_batch_size = 32

id2label = {0: "NRG", 1:"RG"}
label2id = {"NRG": 0, "RG": 1}         

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch32-384")

# define preprocessing and augmentation
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

_train_transforms = Compose(
        [
            Resize(feature_extractor.size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(45),
            GaussianBlur(5, (0.1, 0.2)),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(images):
    return [_train_transforms(img) for img in tqdm(images)]

def val_transforms(images):
    return [_val_transforms(img) for img in tqdm(images)]


def collate_fn(images):
    pixel_values = torch.stack([img["pixel_values"] for img in images])
    labels = torch.LongTensor([img["label"] for img in images])
    return {"pixel_values": pixel_values, "labels": labels}


# define model
class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=10):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch32-384',
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader


if  __name__() == "__main__":
    
    # load data
    labels = pd.read_csv("data/img_info.csv", index_col=0)
    labels.sort_values("shuf_file_number", inplace=True)

    images = []
    for filename in tqdm(os.listdir("data/ods")[:300]):
        filepath = f"data/ods/{filename}"
        img = Image.open(filepath)
        images.append(img)

    images_train = images[:100]
    images_val = images[100:200]
    images_test = images[200:]

    labels_list = labels.labels_int.to_list()
    labels_train = labels_list[:100]
    labels_val = labels_list[100:200]
    labels_test = labels_list[200:]

    # transform data
    train_transformed = train_transforms(images_train)
    val_transformed = val_transforms(images_val)
    test_transformed = val_transforms(images_test)

    train_ds = [{"pixel_values": train_transformed[i], "label": int(labels_train[i])} for i in range(100)]
    val_ds = [{"pixel_values": val_transformed[i], "label": int(labels_val[i])} for i in range(100)]
    test_ds = [{"pixel_values": test_transformed[i], "label": int(labels_test[i])} for i in range(100)]

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=True,
        mode='min'
    )

    model = ViTLightningModule()
    #trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor='validation_loss')], max_epochs=30)
    trainer = Trainer(
        accelerator="auto",
        callbacks=[EarlyStopping(monitor='validation_loss')],
        max_epochs=30,
        auto_lr_find=True,
        deterministic=True,
        max_epochs = 100,
    )
    trainer.fit(model)

    trainer.test()