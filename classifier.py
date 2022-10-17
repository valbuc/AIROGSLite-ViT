import time
from argparse import ArgumentParser
import os
from sys import float_info
from typing import Any, Optional
from PIL import Image
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
from src.transformations import EqualizeTransform, CenterCrop


# id2label = {0: "NRG", 1:"RG"}
# label2id = {"NRG": 0, "RG": 1}
# this is binary classification so we only need 1 class (the positive class)
id2label = {1: "RG"}
label2id = {"RG": 1}


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


def _get_class_labels_df(cls_labels_file):  # e.g. 'data/dev_labels.csv'
    cls_labels = pd.read_csv(cls_labels_file)
    cls_labels['id'] = cls_labels['aimi_id']
    cls_labels['label_num'] = (cls_labels['class'] == 'RG').astype(int)
    cls_labels = cls_labels.drop(columns=['aimi_id', 'class'])
    cls_labels = cls_labels.set_index('id')
    return cls_labels


def _process_idx(idx, obj):
    return idx, obj._get_data(idx)


class ClassifierDataset(Dataset):
    def __init__(self, name, data_dir, df_labels, cache_all=False, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_all = cache_all
        self.data = {}
        self.df_labels = df_labels.reset_index().sort_values(by='filename', ascending=True)
        self.filenames = self.df_labels.filename.to_numpy()
        if self.cache_all:
            print(f"Caching {name} dataset...")
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=8)(delayed(_process_idx)(i, self) for i in tqdm(range(len(self.filenames))))
            for i, data in results:
                self.data[i] = data

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
            if self.transform:
                img = self.transform(img)
            return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self._get_data(idx)
        label = self.df_labels.iloc[idx].label_num
        return img, label


class MyDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(
            title="MyDataModule", description="Class to organize and manage the data"
        )
        parser.add_argument('--data_dir', type=str)  #, e.g. './data/cfp_od_crop_OD_f2.0'
        parser.add_argument('--cls_label_file', default='./data/dev_labels.csv')
        parser.add_argument("--equalize", choices=["no", "yes", "IgnoreBlack"], default="no")
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--od_crop_factor", default=1.5, type=float)
        parser.add_argument("--aug_rot_degrees", default=0, type=float)
        parser.add_argument("--aug_translate", default=0.0, type=float)
        parser.add_argument("--aug_scale", default=0.0, type=float)
        parser.add_argument("--split_num_val_folds", default=5, type=int)
        parser.add_argument("--split_val_fold_idx", default=4, type=int)
        parser.add_argument("--split_test_prop", default=0.0, type=float)
        parser.add_argument("--DATA_MAX_OD_DIAMETER_PROP",
                            default=2.0,
                            help="needs to match data generated with lossless_od_crops_using_yolo_predictions.ipynb")
        parser.add_argument("--DATA_CROP_ENLARGMENT_FACTOR",
                            default=2**0.5 * 1.01,  # so that a rotation of the area of interest will not show an artificial border
                            help="needs to match data generated with  lossless_od_crops_using_yolo_predictions.ipynb")
        return parent_parser

    def __init__(self, args, backbone_transform, backbone_resize):
        super().__init__()

        max_od_diameter_prop = args.DATA_MAX_OD_DIAMETER_PROP
        crop_enlargment_factor = args.DATA_CROP_ENLARGMENT_FACTOR
        crop_factor = args.od_crop_factor / (max_od_diameter_prop * crop_enlargment_factor)
        aug_translate = args.aug_translate / (max_od_diameter_prop * crop_enlargment_factor)

        df_labels = _get_class_labels_df(args.cls_label_file).sort_index(ascending=True)

        equalizer = EqualizeTransform(args)
        self.train_transforms = transforms.Compose([
            equalizer,
            transforms.RandomApply([
                transforms.RandomAffine(degrees=args.aug_rot_degrees,
                                        translate=(aug_translate/2, aug_translate/2),
                                        scale=None)],  # random scaling is done through the cropping
                p=0.9),
            CenterCrop(crop_factor, args.aug_scale/2),
            backbone_resize,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            backbone_transform,
        ])

        self.test_transforms = transforms.Compose([
            equalizer,
            CenterCrop(crop_factor),
            backbone_resize,
            transforms.ToTensor(),
            backbone_transform,
        ])

        datadir_filenames = [fn for fn in os.listdir(args.data_dir) if fn.endswith('.png')]
        fn_df = pd.DataFrame(columns=['filename'], data=datadir_filenames)
        fn_df['id'] = fn_df['filename'].map(lambda x: x[:-4])
        fn_df = fn_df.set_index('id')
        df_labels = pd.merge(fn_df, df_labels, how='left', left_index=True, right_index=True)  # note the left join
        df_labels = df_labels.sort_index(ascending=True)

        devset_len = int(len(df_labels) * (1-args.split_test_prop))
        test_mask = np.zeros(len(df_labels), dtype=bool)
        test_mask[devset_len:-1] = True

        fold_val = np.zeros(len(df_labels), dtype=bool)
        val_cnt = devset_len // args.split_num_val_folds
        if args.split_val_fold_idx < args.split_num_val_folds:
            fold_val[args.split_val_fold_idx * val_cnt:(args.split_val_fold_idx+1) * val_cnt] = True
        else:
            fold_val[args.split_val_fold_idx * val_cnt:devset_len] = True
        fold_train = (~fold_val) & (~test_mask)

        self.train_ds = ClassifierDataset("train", args.data_dir, df_labels=df_labels[fold_train],
                                          cache_all=False, transform=self.train_transforms)
        self.val_ds = ClassifierDataset("validation", args.data_dir, df_labels=df_labels[fold_val],
                                        cache_all=True, transform=self.test_transforms)
        if args.split_test_prop == 0:
            self.test_ds = self.val_ds
        else:
            self.test_ds = ClassifierDataset("test", args.data_dir, df_labels=df_labels[test_mask],
                                             cache_all=False, transform=self.test_transforms)
        self.batch_size = args.batch_size
        self.args = args

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=8, pin_memory=True)

    def test_dataloader(self):
        if self.args.split_test_prop == 0:
            return self.val_dataloader()
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=8)


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
        # operating_point = torch.argwhere(operating_points_with_good_spec).squeeze()[-1]
        # operating_tpr = tpr[operating_point]
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
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--label_smoothing', type=float, default=0.0)
        parser.add_argument('--class_balancing', choices=['focal_loss', 'pos_weight', 'none'], default='none')
        parser.add_argument('--focal_loss_alpha', type=float, default=0.5)
        parser.add_argument('--focal_loss_gamma', type=float, default=2)
        parser.add_argument('--backbone',
                            choices=[
                                'google/vit-base-patch32-384',
                                'microsoft/swin-base-patch4-window12-384-in22k',
                                'microsoft/swin-large-patch4-window12-384-in22k',
                                'tv-224-vit_b_32.IMAGENET1K_V1',
                                'tv-224bic-vit_b_16.IMAGENET1K_SWAG_LINEAR_V1',
                                'tv-384bic-vit_b_16.IMAGENET1K_SWAG_E2E_V1',
                                'tv-224-swin_b.IMAGENET1K_V1',
                                'tv-224-resnext50_32x4d.IMAGENET1K_V2'
                            ], default='hf-vit-384')
        return parent_parser

    def __init__(self, **kwargs):
        super(LitClassifier, self).__init__()
        self.hparams.update(kwargs)
        if self.hparams['backbone'] == 'google/vit-base-patch32-384':
            from transformers import ViTFeatureExtractor, ViTForImageClassification
            self.backbone = ViTForImageClassification.from_pretrained(
                self.hparams['backbone'], num_labels=1, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
            )
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(self.hparams['dropout']),
                self.backbone.classifier
            )
            feature_extractor = ViTFeatureExtractor.from_pretrained(self.hparams['backbone'])
            self.backbone_transform = transforms.Normalize(mean=feature_extractor.image_mean,
                                                           std=feature_extractor.image_std)
            self.backbone_resize = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR)
        elif self.hparams['backbone'] == 'microsoft/swin-base-patch4-window12-384-in22k':
                from transformers import AutoFeatureExtractor, SwinForImageClassification
                self.backbone = SwinForImageClassification.from_pretrained(
                    self.hparams['backbone'], num_labels=1, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
                )
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(self.hparams['dropout']),
                    self.backbone.classifier
                )
                feature_extractor = AutoFeatureExtractor.from_pretrained(self.hparams['backbone'])
                self.backbone_transform = transforms.Normalize(mean=feature_extractor.image_mean,
                                                               std=feature_extractor.image_std)
                self.backbone_resize = transforms.Resize((384, 384),
                                                         interpolation=transforms.InterpolationMode.BILINEAR)
        elif self.hparams['backbone'] == 'microsoft/swin-large-patch4-window12-384-in22k':
                from transformers import AutoFeatureExtractor, SwinForImageClassification
                self.backbone = SwinForImageClassification.from_pretrained(
                    self.hparams['backbone'], num_labels=1, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
                )
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(self.hparams['dropout']),
                    self.backbone.classifier
                )
                feature_extractor = AutoFeatureExtractor.from_pretrained(self.hparams['backbone'])
                self.backbone_transform = transforms.Normalize(mean=feature_extractor.image_mean,
                                                               std=feature_extractor.image_std)
                self.backbone_resize = transforms.Resize((384, 384),
                                                         interpolation=transforms.InterpolationMode.BILINEAR)
        elif self.hparams['backbone'] == 'tv-224-vit_b_32.IMAGENET1K_V1':
            weights = models.ViT_B_32_Weights.IMAGENET1K_V1
            self.backbone = models.vit_b_32(weights=weights)
            self.backbone.heads[0] = nn.Sequential(
                nn.Dropout(self.hparams['dropout']),
                nn.Linear(self.backbone.heads[0].in_features, 1, bias=True)
            )
            self.backbone_transform = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            self.backbone_resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        elif self.hparams['backbone'] == 'tv-224bic-vit_b_16.IMAGENET1K_SWAG_LINEAR_V1':
            # These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
            # weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
            self.backbone = models.vit_b_16(weights=weights)
            self.backbone.heads[0] = nn.Sequential(
                nn.Dropout(self.hparams['dropout']),
                nn.Linear(self.backbone.heads[0].in_features, 1, bias=True)
            )
            self.backbone_transform = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            self.backbone_resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        elif self.hparams['backbone'] == 'tv-384bic-vit_b_16.IMAGENET1K_SWAG_E2E_V1':
            # These weights are learnt via transfer learning by end-to-end fine-tuning the original
            # `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.backbone.heads[0] = nn.Sequential(
                nn.Dropout(self.hparams['dropout']),
                nn.Linear(self.backbone.heads[0].in_features, 1, bias=True)
            )
            self.backbone_transform = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            self.backbone_resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        elif self.hparams['backbone'] == 'tv-224-swin_b.IMAGENET1K_V1':
            self.backbone = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
            self.backbone.head = nn.Sequential(
                nn.Dropout(self.hparams['dropout']),
                nn.Linear(self.backbone.head.in_features, 1, bias=True)
            )
            self.backbone_transform = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            self.backbone_resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        elif self.hparams['backbone'] == 'tv-224-resnext50_32x4d.IMAGENET1K_V2':
            self.backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
            # We change the output layers to make the model compatible to our data
            block_expansion = 4  # from the resnet code
            self.backbone.fc = nn.Sequential(
                nn.Dropout(self.hparams['dropout']),
                nn.Linear(512 * block_expansion, 1, bias=True)
            )
            self.backbone_transform = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            self.backbone_resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

        self.save_hyperparameters()

        if self.hparams['class_balancing'] == 'pos_weight':
            self.class_weights = torch.from_numpy(np.array([1, 13500/1500]))
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights[1])
        elif self.hparams['class_balancing'] == 'focal_loss':
            from torchvision.ops import focal_loss
            def _criterion(logits, target):
                #alpha = 1-1500/13500
                alpha = self.hparams['focal_loss_alpha']
                gamma = self.hparams['focal_loss_gamma']
                return focal_loss.sigmoid_focal_loss(logits, target, alpha, gamma, reduction='mean')
            self.criterion = _criterion
        else:
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
        outputs = self.backbone(pixel_values)
        if type(outputs) == torch.Tensor:
            return outputs
        else:
            return outputs.logits

    def common_step(self, batch, metric_category):
        pixel_values, labels = batch
        logits = self(pixel_values).squeeze()
        y_prob = torch.sigmoid(logits)
        label_smoothing = self.hparams['label_smoothing']

        def _smooth_labels(_labels):
            return _labels * (1 - label_smoothing) + 0.5 * label_smoothing
        loss = self.criterion(logits, _smooth_labels(labels.float()))

        self.log(f'{metric_category}_loss', loss, on_step=True, on_epoch=False)
        self.log(f'{metric_category}_epoch_loss', loss, on_step=False, on_epoch=True)
        metrics = self.metrics[metric_category]
        for name, metric in metrics.items():
            metric_value = metric(y_prob, labels)
            self.log(f"{metric_category}_{name}", metric_value, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        if self.lr_schedulers():
            self.lr_schedulers().step()
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
        def _add_weight_decay(model, weight_decay):
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or ('.bias' in name.lower() or 'norm' in name.lower()):
                    if self.hparams['weight_decay_factor'] > 0:
                        print(f'parameter {name} disabling weight decay')
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        grouped_parameters = _add_weight_decay(self, weight_decay=self.hparams['weight_decay_factor']*self.hparams['lr'])

        optim = None
        if self.hparams['optimizer'] == 'adamw':
            optim = torch.optim.AdamW(grouped_parameters,
                                     lr=self.hparams['lr'],
                                     amsgrad=self.hparams['adamw_amsgrad'])
        elif self.hparams['optimizer'] == 'sgd':
            optim = torch.optim.SGD(grouped_parameters,
                                   lr=self.hparams['lr'],
                                   momentum=self.hparams['sgd_momentum'],
                                   nesterov=self.hparams['sgd_nesterov'])
        assert optim is not None
        ret_dict = {'optimizer': optim}

        if self.hparams['use_lr_scheduler']:
            steps_per_epoch = 15000 * self.hparams['train_prop_end'] / self.hparams['batch_size']
            print(f'Estimated steps_per_epoch {steps_per_epoch}')
            lr_scheduler = get_cosine_schedule_with_warmup(
                optim,
                num_warmup_steps=int(steps_per_epoch*self.hparams['lr_warmup_epochs']),
                num_training_steps=int(steps_per_epoch*self.hparams['lr_training_epochs']))
            ret_dict['lr_scheduler'] = lr_scheduler

        return ret_dict


def cli_main():
    # inspired from https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e
    # https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing#scrollTo=Z92NWT1nB1ZI
    ts_script = time.time()

    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser()
    # program level args
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--es_patience', default=3, type=int)
    parser.add_argument('--es_var', choices=['val_f1', 'val_loss', 'val_partial_auroc', 'val_auroc'], default='val_partial_auroc')
    parser.add_argument('--es_mode', choices=['min', 'max'], default='max')
    parser.add_argument('--use_lr_scheduler', action='store_true')
    parser.add_argument('--lr_training_epochs', default=20, type=int)
    parser.add_argument('--lr_warmup_epochs', default=1, type=int)
    parser = MyDataModule.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    def get_experiment_name(_args):
        d = _args.__dict__
        s = '.'.join([f'{k}_{v}' for k, v in d.items() if k not in ('seed', 'split_val_fold_idx')])
        seed = d["seed"]
        fold = d.get("split_val_fold_idx", 0)
        import hashlib
        hs = hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]
        name = f'h{hs}_s{seed}_f{fold}'
        return name

    args.__dict__['experiment_name'] = get_experiment_name(args)
    print('Experiment name: ', args.experiment_name)

    # always print full weights_summary
    args.weights_summary = 'full'
    # automatically use all available GPUs
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
    args.gpus = -1

    set_seed(args.seed)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="experiment_logs", name=args.experiment_name, default_hp_metric=False
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=args.es_var,
        patience=args.es_patience,
        strict=False,
        verbose=True,
        mode=args.es_mode,
    )

    checkpointing_callback = pl.callbacks.ModelCheckpoint(
        monitor=args.es_var, save_top_k=1, mode=args.es_mode
    )

    model = LitClassifier(**args.__dict__)
    data = MyDataModule(args, model.backbone_transform, model.backbone_resize)

    set_seed(args.seed)  # yes set the seed again to ensure the random state is unaffected

    trainer = pl.Trainer(
        accelerator="auto",
        logger=tb_logger,
        callbacks=[early_stop_callback, checkpointing_callback],
        max_epochs=1000,
        auto_lr_find=False,
        deterministic=True,
        num_sanity_val_steps=0
    )
    # trainer.tune(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trainer.fit(model, data)

    # load the best model for testing
    model.load_from_checkpoint(checkpointing_callback.best_model_path)
    model.eval()

    trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    cli_main()
