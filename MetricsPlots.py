import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.calibration import calibration_curve


class MetricsPlots:

    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels

    def get_auc_roc(self):
        """
        Get the AUC-ROC score

        Return:
             aucroc (float):
                Returns the AUC-ROC score
        """
        aucroc = metrics.roc_auc_score(self.labels, self.predictions)

        return aucroc

    def get_metrics(self):
        """
        Get precision, recall, f1-score and support

        Return:
            metric (dict):
                A dictionary containing the precision, recall, f1-score and support metrics per class

        """
        metric = metrics.classification_report(self.labels, self.predictions, output_dict=True)

        return metric

    def get_sensitivity_95_specificity(self):
        pass

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