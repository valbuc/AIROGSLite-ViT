print("ENtered Script")

import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import torch.nn as nn
import random
import torchmetrics
from torchvision import transforms

# Params
train_batch_size = 128
eval_batch_size = 128

# id2label = {0: "NRG", 1:"RG"}
# label2id = {"NRG": 0, "RG": 1}
id2label = {1:"RG"}
label2id = {"RG": 1}

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch32-384")

# define preprocessing and augmentation
normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)

# _train_transforms = Compose(
#         [
#             #Resize(feature_extractor.size),
#             RandomHorizontalFlip(),
#             #RandomVerticalFlip(),
#             RandomRotation(10),
#             #GaussianBlur(5, (0.1, 0.2)),
#             #ToTensor(),
#             normalize,
#         ]
#     )

_train_transforms = transforms.Compose([transforms.ToTensor(),
                                        normalize,
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([
                                            transforms.RandomAffine(degrees=10, translate=None, scale=(1.0, 1.2))],
                                            p=0.5)
                                        ])

_val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.RandomHorizontalFlip(),
])

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
                img = Image.open(filepath).convert('RGB')  # convert forces the image to load in the main process
                # ...other, since PIL uses lazy loading, would cause the image to be loaded in the dataloader
                # worker process but PIL has issues with multiprocessing
                self.data[i] = img

    def __len__(self):
        return len(self.filenames)

    def _get_data(self, idx):
        cached_val = self.data.get(idx, None)
        if cached_val is not None:
            return cached_val
        else:
            fn = self.filenames[idx]
            filepath = os.path.join(self.data_dir, fn)
            img = Image.open(filepath).convert('RGB')
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


# define model
class ViTLightningModule(pl.LightningModule):
    def __init__(self):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch32-384',
            num_labels=1,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        self.hparams['lr'] = 1e-3
        self.criterion = nn.BCEWithLogitsLoss()

        self.metrics = {}
        for metric_cat in ['train', 'val', 'test']:
            self.metrics[metric_cat] = {
                'acc': torchmetrics.Accuracy(),
                'f1': torchmetrics.F1Score(multiclass=False),
                'prec': torchmetrics.Precision(multiclass=False),
                'rec': torchmetrics.Recall(multiclass=False)
            }

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, metric_category):
        pixel_values, labels = batch
        logits = self(pixel_values).squeeze()
        y_prob = torch.sigmoid(logits)
        loss = self.criterion(logits, labels.float())

        self.log(f'{metric_category}_loss', loss, on_step=True, on_epoch=False)
        self.log(f'{metric_category}_epoch_loss', loss, on_step=False, on_epoch=True)
        metrics = self.metrics[metric_category]
        for name, metric in metrics.items():
            metric_value = metric.to(labels.device)(y_prob, labels)
            self.log(f"{metric_category}_{name}", metric_value, on_step=False, on_epoch=True)

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, 'test')
        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['lr'], amsgrad=True)
        #return torch.optim.SGD(self.parameters(), lr=self.hparams['lr'], momentum=0.9, nesterov=True)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader


if __name__ == "__main__":

    set_seed(123)

    # load data
    labels = pd.read_csv("./data/ods/img_info.csv", index_col=0)
    n_images = labels.shape[0]
    #n_images = 3000

    train_end_idx = int(n_images * 0.8)
    val_end_idx = int(n_images * 0.99)

    train_ds = ClassifierDataset('./data/ods', cache_all=True, start_idx=0, end_idx=train_end_idx, transform=_train_transforms)
    val_ds = ClassifierDataset('./data/ods', cache_all=True, start_idx=train_end_idx, end_idx=val_end_idx, transform=_val_transforms)
    test_ds = ClassifierDataset('./data/ods', start_idx=val_end_idx, end_idx=n_images, transform=_val_transforms)

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=train_batch_size, num_workers=6, persistent_workers=True)
    val_dataloader = DataLoader(val_ds, batch_size=eval_batch_size, num_workers=6, persistent_workers=True)
    test_dataloader = DataLoader(test_ds, batch_size=eval_batch_size, num_workers=6, persistent_workers=True)

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_f1',
        patience=5,
        strict=False,
        verbose=True,
        mode='max'
    )

    model = ViTLightningModule()
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=early_stop_callback,
        max_epochs=100,
        auto_lr_find=True,
        deterministic=True,
    )
    #trainer.tune(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trainer.fit(model)

    trainer.test()