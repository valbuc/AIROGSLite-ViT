import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from PIL import Image
import random
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from src.transformations import EqualizeTransform, CenterCrop
from src.model import LitClassifier


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


class ClassifierDataset(Dataset):
    def __init__(self, data_dir, filenames, labels, cache_all=False, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.cache_all = cache_all
        self.data = {}
        self.filenames = filenames
        self.labels = labels

    def __len__(self):
        return len(self.filenames)

    def _get_data(self, idx):
        if self.cache_all:
            cached_val = self.data.get(idx, None)
            if cached_val is not None:
                return cached_val
        fn = self.filenames[idx]
        filepath = os.path.join(self.data_dir, fn)
        img = Image.open(filepath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.cache_all:
            self.data[idx] = img
        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self._get_data(idx)
        label = np.nan
        if self.labels is not None:
            label = self.labels[idx]
        return img, label, self.filenames[idx]


def _get_train_val_test_split_masks(data_cnt, test_prop, fold_cnt, fold_idx):
    assert fold_idx < fold_cnt
    devset_len = int(data_cnt * (1 - test_prop))
    test_mask = np.zeros(data_cnt, dtype=bool)
    test_mask[devset_len:-1] = True

    fold_val = np.zeros_like(test_mask)
    val_cnt = devset_len // fold_cnt
    fold_val[fold_idx * val_cnt:(fold_idx + 1) * val_cnt] = True

    fold_train = (~fold_val) & (~test_mask)

    return fold_train, fold_val, test_mask


class MyDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(
            title="MyDataModule", description="Class to organize and manage the data"
            
        )
        parser.add_argument('--data_dir', type=str, #, e.g. './data/cfp_od_crop_OD_f2.0'
                            help='location of the images used for training, validation and test of the classifier')
        parser.add_argument('--predict_data_dir', type=str, #, e.g. './data/cfp_od_crop_OD_f2.0'
                            help='location of the images used for prediction (competition test images)')
        parser.add_argument('--cls_label_file', default='./data/dev_labels.csv',
                            help='the target labels used for training')
        parser.add_argument("--aug_hist_equalize", choices=["no", "yes", "IgnoreBlack"], default="no",
                            help='data augmentation: histogram equalization')
        parser.add_argument("--batch_size", default=16, type=int,
                            help='batch size')
        parser.add_argument("--no_crop_resize", action='store_true',
                            help='flag to indicate that no center crop and resize should be performed (e.g. when using the whole CFP images)')
        parser.add_argument("--od_crop_factor", default=1.5, type=float,
                            help='data processing step: crop size as a proportion of the optic disk')
        parser.add_argument("--aug_rot_degrees", default=0, type=float,
                            help='data augmentation: range of rotation angles')
        parser.add_argument("--aug_translate", default=0.0, type=float,
                            help='data augmentation: range of horizontal and vertical translation')
        parser.add_argument("--aug_scale", default=0.0, type=float,
                            help='data augmentation: change the crop size so that image appeares scaled')
        parser.add_argument("--split_num_val_folds", default=5, type=int,
                            # help='validation config: which validation fold to use out of split_val_fold_idx possible ('
                            #      'e.g. if split_val_fold_idx is 5 then a split_num_val_folds of 0 means that the first '
                            #      '20% of the dev-set images will be used as a validation set'
                            )
        parser.add_argument("--split_val_fold_idx", default=4, type=int,
                            help='validation config: 100/split_val_fold_idx is the percentage of non-test data to use for validation')
        parser.add_argument("--split_test_prop", default=0.0, type=float,
                            help='proportion of data to use as a test set')
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
        if args.no_crop_resize:
            self.train_transforms = transforms.Compose([
                equalizer,
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=args.aug_rot_degrees,
                                            translate=(args.aug_translate / 2, args.aug_translate / 2),
                                            scale=None)],  # random scaling is done through the cropping
                    p=0.66),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                backbone_transform,
            ])

            self.test_transforms = transforms.Compose([
                equalizer,
                transforms.ToTensor(),
                backbone_transform,
            ])
        else:
            self.train_transforms = transforms.Compose([
                equalizer,
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=args.aug_rot_degrees,
                                            translate=(aug_translate/2, aug_translate/2),
                                            scale=None)],  # random scaling is done through the cropping
                    p=0.66),
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

        # always shuffle before, to be sure that the data is not biased in any way
        rng = np.random.default_rng(seed=123)
        shuffled_order = rng.permuted(np.arange(len(df_labels), dtype=int))
        df_labels = df_labels.iloc[shuffled_order]
        train_mask, val_mask, test_mask = _get_train_val_test_split_masks(
            len(df_labels), args.split_test_prop, args.split_num_val_folds, args.split_val_fold_idx)

        self.train_ds = ClassifierDataset(args.data_dir, cache_all=False, transform=self.train_transforms,
                                          filenames=df_labels[train_mask].filename.to_numpy(),
                                          labels=df_labels[train_mask].label_num.to_numpy())
        self.val_ds = ClassifierDataset(args.data_dir, cache_all=True, transform=self.test_transforms,
                                        filenames=df_labels[val_mask].filename.to_numpy(),
                                        labels=df_labels[val_mask].label_num.to_numpy())
        if args.split_test_prop == 0:
            self.test_ds = self.val_ds
        else:
            self.test_ds = ClassifierDataset(args.data_dir, cache_all=False, transform=self.test_transforms,
                                             filenames=df_labels[test_mask].filename.to_numpy(),
                                             labels=df_labels[test_mask].label_num.to_numpy())

        if args.predict_data_dir:
            datadir_filenames = sorted([fn for fn in os.listdir(args.predict_data_dir) if fn.endswith('.png')])
            self.predict_ds = ClassifierDataset(args.predict_data_dir, cache_all=False, transform=self.test_transforms,
                                                filenames=datadir_filenames, labels=None)
        self.batch_size = args.batch_size
        self.args = args

        self.train_data_loader = DataLoader(
            self.train_ds, shuffle=True, batch_size=self.batch_size,
            num_workers=8, persistent_workers=True, pin_memory=True,
            drop_last=True)
        self.val_data_loader = DataLoader(
            self.val_ds, shuffle=False, batch_size=self.batch_size,
            num_workers=1, persistent_workers=True, pin_memory=True)
        self.test_data_loader = DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def predict_dataloader(self):
        return DataLoader(self.predict_ds, shuffle=False, batch_size=self.batch_size, num_workers=8)

    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.val_data_loader

    def test_dataloader(self):
        return self.test_data_loader


def cli_main():
    # inspired from https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e
    # https://colab.research.google.com/drive/17CtsJtGCjp4YkykIpIoY0Kdb9nCadeFT?usp=sharing#scrollTo=Z92NWT1nB1ZI
    ts_script = time.time()

    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # program level args
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--es_patience', default=3, type=int,
                        help='early stopping: how many (half) epochs to check without an improvement in the es metric')
    parser.add_argument('--es_var', choices=['val_f1', 'val_loss', 'val_partial_auroc', 'val_auroc'], default='val_partial_auroc',
                        help='early stopping: which metric to use for early stopping (stop training when the metric doesn\'t improve on the validations set')
    parser.add_argument('--es_mode', choices=['min', 'max'], default='max',
                        help='early stopping: ')
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help='flag indicating whether a learning rate scheduler (cosine schedule with linear warmup) should be used')
    parser.add_argument('--lr_training_epochs', default=20, type=int,
                        help='lr has decayed to 0 after these many epochs (only applicable in conjunction with use_lr_scheduler)')
    parser.add_argument('--lr_warmup_epochs', default=1, type=int,
                        help='lr increases from 0 to the configured value in this many epochs')
    parser.add_argument('--tensorboard_log_dir', default='experiment_logs', type=str,
                        help='location where to save the metrics which can be loaded by tensorboard')

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
        save_dir=args.tensorboard_log_dir, name=args.experiment_name, default_hp_metric=False
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
        num_sanity_val_steps=0,
        val_check_interval=0.5  # the model converges quickly, so the validation should be done more frequently for better model selection
    )

    trainer.fit(model, data)

    set_seed(args.seed)  # yes set the seed again to ensure the random state is unaffected before testing

    # load the best model for testing
    model.load_from_checkpoint(checkpointing_callback.best_model_path)
    model.eval()

    trainer.test(model=model, datamodule=data)

    if args.predict_data_dir:
        dfs = []
        predictions = trainer.predict(model, data.predict_dataloader())
        for preds, filenames in predictions:
            df = pd.DataFrame(data={'filename': filenames, 'predictions': preds})
            dfs.append(df)
        df_test_res = pd.concat(dfs).sort_values(by='filename', ascending=True)
        test_res_file = f'{args.tensorboard_log_dir}/submission_predictions_{args.experiment_name}.csv'
        df_test_res.to_csv(test_res_file)


if __name__ == "__main__":
    cli_main()
