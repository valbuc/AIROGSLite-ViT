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


class DataLoaderHolder(object):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(
            title="DataLoader", description="Class for loading the data to predict on"
        )
        parser.add_argument('--data_dir', type=str, #, e.g. './data/cfp_od_crop_OD_f2.0'
                            help='location of the images used for training, validation and test of the classifier', required=True)
        parser.add_argument("--batch_size", default=64, type=int,
                            help='batch size')
        parser.add_argument("--aug_hist_equalize", choices=["no", "yes", "IgnoreBlack"], default="no",
                            help='histogram equalization')
        parser.add_argument("--no_crop_resize", action='store_true',
                            help='flag to indicate that no center crop and resize should be performed (e.g. when using the whole CFP images)')
        parser.add_argument("--od_crop_factor", default=1.5, type=float,
                            help='data processing step: crop size as a proportion of the optic disk')
        parser.add_argument("--DATA_MAX_OD_DIAMETER_PROP", type=float,
                            default=2.0,
                            help="needs to match data generated with lossless_od_crops_using_yolo_predictions.ipynb")
        parser.add_argument("--DATA_CROP_ENLARGMENT_FACTOR", type=float,
                            default=2**0.5 * 1.01,  # so that a rotation of the area of interest will not show an artificial border
                            help="needs to match data generated with  lossless_od_crops_using_yolo_predictions.ipynb")
        return parent_parser

    def __init__(self, args, backbone_transform, backbone_resize):

        max_od_diameter_prop = args.DATA_MAX_OD_DIAMETER_PROP
        crop_enlargment_factor = args.DATA_CROP_ENLARGMENT_FACTOR
        crop_factor = args.od_crop_factor / (max_od_diameter_prop * crop_enlargment_factor)

        equalizer = EqualizeTransform(args)
        if args.no_crop_resize:
            self.test_transforms = transforms.Compose([
                equalizer,
                transforms.ToTensor(),
                backbone_transform,
            ])
        else:
            self.test_transforms = transforms.Compose([
                equalizer,
                CenterCrop(crop_factor),
                backbone_resize,
                transforms.ToTensor(),
                backbone_transform,
            ])

        datadir_filenames = sorted([fn for fn in os.listdir(args.data_dir) if fn.endswith('.png')])
        self.pred_ds = ClassifierDataset(args.data_dir, cache_all=False, transform=self.test_transforms,
                                         filenames=datadir_filenames,
                                         labels=None)
        self.batch_size = args.batch_size
        self.args = args

        self.prediction_data_loader = DataLoader(self.pred_ds, shuffle=False, batch_size=self.batch_size, num_workers=8)


def cli_main():
    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # program level args
    parser.add_argument('--model_checkpoint', type=str,
                        help='path to the model checkpoint used for prediction', required=True)
    parser.add_argument('--out_file_prefix', type=str,
                        help='', default='prediction')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed')

    parser = DataLoaderHolder.add_argparse_args(parser)
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    args.__dict__['experiment_name'] = args.model_checkpoint

    # always print full weights_summary
    args.weights_summary = 'full'
    # automatically use all available GPUs
    # https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices
    args.gpus = -1

    set_seed(args.seed)

    model = LitClassifier.load_from_checkpoint(args.model_checkpoint)
    data = DataLoaderHolder(args, model.backbone_transform, model.backbone_resize)

    dfs = []
    trainer = pl.Trainer(accelerator="auto")
    predictions = trainer.predict(model=model, dataloaders=data.prediction_data_loader)
    for preds, filenames in predictions:
        df = pd.DataFrame(data={'filename': filenames, 'predictions': preds})
        dfs.append(df)

    df_test_res = pd.concat(dfs).sort_values(by='filename', ascending=True)
    print(os.path.split(args.model_checkpoint))
    predictions_file = args.out_file_prefix + '_' + \
                       os.path.split(args.data_dir)[1] + \
                       os.path.split(args.model_checkpoint)[-1] + '.csv'
    df_test_res.to_csv(predictions_file)
    print(f'Written test predictions to {predictions_file}')


if __name__ == "__main__":
    cli_main()
