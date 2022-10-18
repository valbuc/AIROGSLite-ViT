from sys import float_info
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from torchvision import transforms, models
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup


id2label = {1: "RG"}
label2id = {"RG": 1}


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
        parser.add_argument('--lr', type=float, default=5e-5,
                            help='learning rate')
        parser.add_argument('--optimizer', choices=['adamw', 'sgd'], default='adamw',
                            help='optimizer')
        parser.add_argument('--weight_decay_factor', type=float, default=0,
                            help='optimizer parameter')
        parser.add_argument('--sgd_nesterov', type=bool, default=False,
                            help='SGD optimizer parameter')
        parser.add_argument('--sgd_momentum', type=float, default=0.9,
                            help='SGD optimizer parameter')
        parser.add_argument('--adamw_amsgrad', type=bool, default=False,
                            help='AdamW optimmizer parameter')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='dropout regularization on the last linear layer of the model')
        parser.add_argument('--label_smoothing', type=float, default=0.0,
                            help='apply label smoothing during training. 0.1 means that target 0 become 0.05 and target 1 becomes 0.95.')
        parser.add_argument('--class_balancing', choices=['focal_loss', 'pos_weight', 'none'], default='none',
                            help='what method to use for addressing class imbalance')
        parser.add_argument('--focal_loss_alpha', type=float, default=0.5,
                            help='applicable only if class_balancing is focal_loss')
        parser.add_argument('--focal_loss_gamma', type=float, default=2,
                            help='applicable only if class_balancing is focal_loss')
        parser.add_argument('--backbone',
                            choices=[
                                'google/vit-base-patch32-384',
                                'microsoft/swin-base-patch4-window12-384-in22k',
                                'microsoft/swin-large-patch4-window12-384-in22k',
                                'tv-224-vit_b_32.IMAGENET1K_V1',
                                'tv-224vit_b_16.IMAGENET1K_SWAG_LINEAR_V1',
                                'tv-384vit_b_16.IMAGENET1K_SWAG_E2E_V1',
                                'tv-224-swin_b.IMAGENET1K_V1',
                                'tv-224-resnext50_32x4d.IMAGENET1K_V2'
                            ], default='tv-224-swin_b.IMAGENET1K_V1',
                            help='choice of model and pretrained weigths for transfer learning')
        return parent_parser

    def __init__(self, **kwargs):
        super(LitClassifier, self).__init__()
        self.hparams.update(kwargs)
        if self.hparams['backbone'] == 'google/vit-base-patch32-384':
            from transformers import ViTFeatureExtractor, ViTForImageClassification
            self.backbone = ViTForImageClassification.from_pretrained(
                self.hparams['backbone'], num_labels=1, id2label=id2label, label2id=label2id,
                ignore_mismatched_sizes=True
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
                self.hparams['backbone'], num_labels=1, id2label=id2label, label2id=label2id,
                ignore_mismatched_sizes=True
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
                self.hparams['backbone'], num_labels=1, id2label=id2label, label2id=label2id,
                ignore_mismatched_sizes=True
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
        elif self.hparams['backbone'] == 'tv-224vit_b_16.IMAGENET1K_SWAG_LINEAR_V1':
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
        elif self.hparams['backbone'] == 'tv-384vit_b_16.IMAGENET1K_SWAG_E2E_V1':
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
            self.backbone_resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
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
            self.class_weights = torch.from_numpy(np.array([1, 13500 / 1500]))
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights[1])
        elif self.hparams['class_balancing'] == 'focal_loss':
            from torchvision.ops import focal_loss
            def _criterion(logits, target):
                # alpha = 1-1500/13500
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
                # 'prec': torchmetrics.Precision(multiclass=False),
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
        pixel_values, labels, _ = batch
        logits = self(pixel_values).squeeze(dim=1)
        y_prob = torch.sigmoid(logits)
        label_smoothing = self.hparams['label_smoothing']

        def _smooth_labels(_labels):
            return _labels * (1 - label_smoothing) + 0.5 * label_smoothing

        loss = self.criterion(logits, _smooth_labels(labels))

        self.log(f'{metric_category}_loss', loss, on_step=True, on_epoch=False, batch_size=len(labels))
        self.log(f'{metric_category}_epoch_loss', loss, on_step=False, on_epoch=True, batch_size=len(labels))
        metrics = self.metrics[metric_category]
        for name, metric in metrics.items():
            metric_value = metric(y_prob, labels)
            self.log(f"{metric_category}_{name}", metric_value, on_step=False, on_epoch=True, batch_size=len(labels))

        return loss, y_prob, labels

    def training_step(self, batch, batch_idx):
        if self.lr_schedulers():
            self.lr_schedulers().step()
        loss, _, _ = self.common_step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, 'val')
        # return loss

    def test_step(self, batch, batch_idx):
        _, preds, labels = self.common_step(batch, 'test')
        _, _, filenames = batch
        return preds.cpu().numpy(), labels.cpu().numpy(), filenames

    def test_epoch_end(self, outputs):
        dfs = []
        for preds, labels, filenames in outputs:
            df = pd.DataFrame(data={'filename': filenames, 'predictions': preds, 'labels': labels})
            dfs.append(df)
        df_test_res = pd.concat(dfs).sort_values(by='filename', ascending=True)
        test_res_file = f'{self.hparams.tensorboard_log_dir}/predictions_{self.hparams.experiment_name}.csv'
        df_test_res.to_csv(test_res_file)
        print(f'Written test predictions to {test_res_file}')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return torch.sigmoid(self(batch[0]).squeeze(dim=1)).cpu().numpy(), batch[2]  # pred, filename

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

        grouped_parameters = _add_weight_decay(self,
                                               weight_decay=self.hparams['weight_decay_factor'] * self.hparams['lr'])

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
                num_warmup_steps=int(steps_per_epoch * self.hparams['lr_warmup_epochs']),
                num_training_steps=int(steps_per_epoch * self.hparams['lr_training_epochs']))
            ret_dict['lr_scheduler'] = lr_scheduler

        return ret_dict
