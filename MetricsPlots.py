import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.calibration import calibration_curve


class MetricsPlots:

    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels

    def get_partial_auc_roc(self):
        """
        Get the AUC-ROC score

        Return:
             p_aucroc (float):
                Returns the AUC-ROC score
        """
        min_spec = 0.9
        p_aucroc = metrics.roc_auc_score(self.labels, self.predictions, max_fpr=(1 - min_spec))

        return p_aucroc

    def get_metrics(self):
        """
        Get precision, recall, f1-score and support

        Return:
            metric (dict):
                A dictionary containing the precision, recall, f1-score and support metrics per class

        """
        metric = metrics.classification_report(self.labels, self.predictions, output_dict=True)

        return metric

    def screening_sens_at_spec(self, at_spec, eps=sys.float_info.epsilon):
        y_true = self.labels
        y_pred = self.predictions

        fpr, tpr, threshes = metrics.roc_curve(y_true, y_pred, drop_intermediate=False)
        spec = 1 - fpr

        operating_points_with_good_spec = spec >= (at_spec - eps)
        max_tpr = tpr[operating_points_with_good_spec][-1]

        operating_point = np.argwhere(operating_points_with_good_spec).squeeze()[-1]
        operating_tpr = tpr[operating_point]

        assert max_tpr == operating_tpr or (
                np.isnan(max_tpr) and np.isnan(operating_tpr)), f'{max_tpr} != {operating_tpr}'
        assert max_tpr == max(tpr[operating_points_with_good_spec]) or (
                np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
            f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'

        return max_tpr

    def plot_auc_roc(self):
        """
        Plot the AUC-ROC curve
        """

        # Plot roc curve
        auc_fpr, auc_tpr, auc_thresholds = metrics.roc_curve(self.labels, self.predictions)
        y, x = calibration_curve(self.labels, self.predictions, n_bins=10)

        # plot the roc curve for the model
        plt.figure()
        plt.ylim(0., 1.0)
        plt.xlim(0., 1.0)
        plt.plot(auc_fpr, auc_tpr, marker='.', color='darkorange', label="AUC-ROC")
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.plot(y, x, marker='^', linestyle="", markersize=7, color='darkorange', label="calibration curve")

        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()

        plt.show()