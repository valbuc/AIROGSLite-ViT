print("ENtered Script")

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification, AdamW
import pandas as pd
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import pytorch_lightning as pl
import torch.nn as nn
import random
from torchvision.transforms import (Compose,
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    GaussianBlur,
                                    RandomRotation, #  rotation
                                    Resize, 
                                    ToTensor)


# Params
train_batch_size = 64
eval_batch_size = 64

id2label = {0: "NRG", 1:"RG"}
label2id = {"NRG": 0, "RG": 1}         

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch32-384")

# define preprocessing and augmentation
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

_train_transforms = Compose(
        [
            Resize(feature_extractor.size),
            RandomHorizontalFlip(),
            #RandomVerticalFlip(),
            RandomRotation(10),
            #GaussianBlur(5, (0.1, 0.2)),
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

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.RandomState(seed)
    random.seed(seed)
    pl.utilities.seed.seed_everything(seed)

class ClassifierDataset(Dataset):
    def __init__(self, data_dir, start_idx, end_idx, cache_all=False, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_all = cache_all
        self.data = {}
        labels = pd.read_csv(os.path.join(data_dir, "img_info.csv"), index_col=0)
        self.df_labels = labels.set_index("new_file").sort_index(ascending=True)
        self.df_labels = self.df_labels.iloc[start_idx:end_idx, :]
        self.filenames = self.df_labels.index.to_numpy()

        if self.cache_all:
            for i, fn in enumerate(self.filenames):
                filepath = os.path.join(self.data_dir, fn)
                img = Image.open(filepath)
                self.data[i] = img
                # forcing the image to be loaded (PIL uses lazy loading which is messed up in multiprocessing)
                px = img.load()
                px[img.size[0]-1, img.size[1]-1]

    def __len__(self):
        return len(self.filenames)

    def _get_data(self, idx):
        cached_val = self.data.get(idx, None)
        if cached_val:
            return cached_val
        else:
            fn = self.filenames[idx]
            filepath = os.path.join(self.data_dir, fn)
            img = Image.open(filepath)
            return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self._get_data(idx)

        # apply transforms
        if self.transform:
            img = self.transform(img)

        label = int(self.df_labels.loc[self.filenames[idx]].labels_int)
        return img, label

def train_transforms(images):
    return [_train_transforms(img) for img in tqdm(images)]

def val_transforms(images):
    return [_val_transforms(img) for img in tqdm(images)]


# def collate_fn(images):
#     pixel_values = torch.stack([img["pixel_values"] for img in images])
#     labels = torch.LongTensor([img["label"] for img in images])
#     return {"pixel_values": pixel_values, "labels": labels}

def collate_fn(images):
    pixel_values = torch.stack([img[0] for img in images])
    labels = torch.LongTensor([img[1] for img in images])
    return {"pixel_values": pixel_values, "labels": labels}

# define model
class ViTLightningModule(pl.LightningModule):
    def __init__(self):
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


if __name__ == "__main__":
    
    # load data
    labels = pd.read_csv("./data/ods/img_info.csv", index_col=0)
    n_images = labels.shape[0]
    #n_images = 300

    train_end_idx = int(n_images * 0.6)
    val_end_idx = int(n_images * 0.8)

    train_ds = ClassifierDataset('./data/ods', cache_all=True, start_idx=0, end_idx=train_end_idx, transform=_train_transforms)
    val_ds = ClassifierDataset('./data/ods', cache_all=True, start_idx=train_end_idx, end_idx=val_end_idx, transform=_val_transforms)
    test_ds = ClassifierDataset('./data/ods', start_idx=val_end_idx, end_idx=n_images, transform=_val_transforms)

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=4, persistent_workers=True)

    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='validation_loss',
        patience=5,
        strict=False,
        verbose=True,
        mode='min'
    )

    model = ViTLightningModule()
    #trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor='validation_loss')], max_epochs=30)
    trainer = pl.Trainer(
        accelerator="auto",
        #callbacks=[EarlyStopping(monitor='validation_loss', patience=10)],
        callbacks=early_stop_callback,
        max_epochs=100,
        auto_lr_find=True,
        deterministic=True,
    )
    trainer.fit(model)

    trainer.test()