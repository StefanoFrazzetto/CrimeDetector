from enum import Enum


class MetricType(Enum):
    ACCURACY = 'Accuracy'
    PRECISION = 'Precision'
    RECALL = 'Recall'

    F05 = 'F0.5'
    F1 = 'F1'
    F2 = 'F2'
    F3 = 'F3'

    AUC = 'AUC'
    FPR = 'FPR'
    TPR = 'TPR'
    ROC = 'ROC'
    THRESHOLDS = 'Thresholds'

    MCC = 'Matthews Correlation Coefficient'
    CONFUSION_MATRIX = 'Confusion Matrix'

    def __str__(self):
        return self.value
