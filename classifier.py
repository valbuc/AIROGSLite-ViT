import time
from argparse import ArgumentParser
import os
from sys import float_info
from PIL import Image
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import ViTFeatureExtractor, ViTForImageClassification


# id2label = {0: "NRG", 1:"RG"}
# label2id = {"NRG": 0, "RG": 1}
# this is binary classification so we only need 1 class (the positive class)
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
        labels = pd.read_csv(os.path.join(data_dir, "img_info_with_labels.csv"), index_col=0)
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

class MyDataModule(pl.LightningDataModule):

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(
            title="MyDataModule",
            description="Class to organize and manage the data"
        )
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--use_validation_set_for_test', action="store_true")
        parser.add_argument('--train_prop_end', default=0.6, type=float)
        parser.add_argument('--val_prop_end', default=0.8, type=float)
        return parent_parser

    def __init__(self, args):
        super().__init__()

        self.train_transforms = transforms.Compose([transforms.ToTensor(),
                                                    normalize,
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomApply([
                                                        transforms.RandomAffine(degrees=10, translate=None,
                                                                                scale=(1.0, 1.2))],
                                                        p=0.5)
                                                    ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            transforms.RandomHorizontalFlip(),
        ])

        self.labels = pd.read_csv(os.path.join(args.data_dir, "img_info_with_labels.csv"), index_col=0)
        n_images = self.labels.shape[0]
        #n_images = 300

        train_end_idx = int(n_images * args.train_prop_end)
        val_end_idx = int(n_images * args.val_prop_end)

        self.train_ds = ClassifierDataset(args.data_dir, cache_all=True,
                                          start_idx=0, end_idx=train_end_idx,
                                          transform=self.train_transforms)
        self.val_ds = ClassifierDataset(args.data_dir, cache_all=True,
                                        start_idx=train_end_idx, end_idx=val_end_idx,
                                        transform=self.test_transforms)
        if args.use_validation_set_for_test:
            self.test_ds = self.val_ds
        else:
            self.test_ds = ClassifierDataset(args.data_dir,
                                             start_idx=val_end_idx, end_idx=n_images,
                                             transform=self.test_transforms)
        self.batch_size = args.batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size,
                          num_workers=6, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=6, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          num_workers=6, persistent_workers=True)

from typing import Any, Optional

class SensAtSpec(torchmetrics.Metric):
    r"""
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        at_specificity: Optional[float] = 0.95,
        eps=float_info.epsilon,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.at_specificity = at_specificity
        self.epsilon = eps
        self.roc_metric = torchmetrics.ROC()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        self.roc_metric.update(preds, target)

    def compute(self) -> torch.Tensor:
        fpr, tpr, threshes = self.roc_metric.compute()
        spec = 1 - fpr
        operating_points_with_good_spec = spec >= (self.at_specificity - self.epsilon)
        max_tpr = tpr[operating_points_with_good_spec][-1]
        #operating_point = torch.argwhere(operating_points_with_good_spec).squeeze()[-1]
        #operating_tpr = tpr[operating_point]
        # assert max_tpr == operating_tpr or (np.isnan(max_tpr) and np.isnan(operating_tpr)), f'{max_tpr} != {operating_tpr}'
        # assert max_tpr == max(tpr[operating_points_with_good_spec]) or (np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
        #     f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'
        return max_tpr

# define model
class LitClassifier(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(
            title="LitClassifier",
            description="Classifier with multiple possible backbones"
        )
        parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
        parser.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw')
        parser.add_argument('--weight_decay_factor', type=float, default=0)
        parser.add_argument('--sgd_nesterov', type=bool, default=False)
        parser.add_argument('--sgd_momentum', type=float, default=0.9)
        parser.add_argument('--adamw_amsgrad', type=bool, default=False)
        return parent_parser

    def __init__(self, **kwargs):
        super(LitClassifier, self).__init__()
        self.backbone = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch32-384',
            num_labels=1,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.hparams.update(kwargs)
        self.save_hyperparameters()

        self.criterion = nn.BCEWithLogitsLoss()

        self.metrics = {}
        for metric_cat in ['train', 'val', 'test']:
            self.metrics[metric_cat] = {
                'acc': torchmetrics.Accuracy(),
                'f1': torchmetrics.F1Score(multiclass=False),
                #'prec': torchmetrics.Precision(multiclass=False),
                'sensitivity': torchmetrics.Recall(multiclass=False),
                'specificity': torchmetrics.Specificity(multiclass=False),
                'auroc': torchmetrics.AUROC(),
                # one of the metrics for the challenge: partial auroc (90-100% specificity)
                # since specificity = 1 - false positive rate <=> partial auroc for 0-10 false positive rate
                # <=> max_fpr = 0.1
                'partial_auroc': torchmetrics.AUROC(max_fpr=0.1),
                'sens_at_95_spec': SensAtSpec(at_specificity=0.95)
            }
            for k, v in self.metrics[metric_cat].items():
                self.register_module(f'metric_{metric_cat}_{k}', v)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
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
            metric_value = metric(y_prob, labels)
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

    def on_train_start(self):
        # Ensuring that the test metrics are logged also in the hyperparameters tab
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html
        self.logger.log_hyperparams(self.hparams, {f'test_{k}': 0 for k in self.metrics['test'].keys()})

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        if self.hparams['optimizer'] == 'adamw':
            return torch.optim.AdamW(self.parameters(),
                                     lr=self.hparams['lr'],
                                     weight_decay=self.hparams['weight_decay_factor']*self.hparams['lr'],
                                     amsgrad=self.hparams['adamw_amsgrad'])
        elif self.hparams['optimizer'] == 'sgd':
            return torch.optim.SGD(self.parameters(),
                                   lr=self.hparams['lr'],
                                   momentum=self.hparams['sgd_momentum'],
                                   nesterov=self.hparams['sgd_nesterov'])


def cli_main():
    # inspired from https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e
    # https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing#scrollTo=Z92NWT1nB1ZI
    ts_script = time.time()

    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser()
    # program level args
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--experiment_name', default='no_experiment', type=str)
    parser.add_argument('--es_patience', default=5, type=int)
    parser.add_argument('--es_var', choices=['val_f1', 'val_loss', 'val_partial_auroc', 'val_auroc'], default='val_partial_auroc')
    parser.add_argument('--es_mode', choices=['min', 'max'], default='max')
    parser.add_argument('--data_dir', default='./data/ods')
    parser = MyDataModule.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # always print full weights_summary
    args.weights_summary = 'full'
    # automatically use all available GPUs
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
    args.gpus = -1

    set_seed(args.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='experiment_logs',
        name=args.experiment_name,
        default_hp_metric=False)

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=args.es_var,
        patience=args.es_patience,
        strict=False,
        verbose=True,
        mode=args.es_mode,
    )

    checkpointing_callback = pl.callbacks.ModelCheckpoint(monitor=args.es_var, save_top_k=1, mode=args.es_mode)

    data = MyDataModule(args)
    model = LitClassifier(**args.__dict__)

    trainer = pl.Trainer(
        accelerator="auto",
        logger=tb_logger,
        callbacks=[early_stop_callback, checkpointing_callback],
        max_epochs=100,
        auto_lr_find=False,
        deterministic=True,
    )
    #trainer.tune(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trainer.fit(model, data)

    # load the best model for testing
    model.load_from_checkpoint(checkpointing_callback.best_model_path)
    model.eval()

    trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    cli_main()
