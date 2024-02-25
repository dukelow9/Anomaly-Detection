import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plotROCAUC(labels, reconstructionErrors):
    fpr, tpr, thresholds = roc_curve(labels, reconstructionErrors)
    ROCAUC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='lime', lw=2, label=f'AUC = {ROCAUC:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    
plotROCAUC(y, reconstructionErrors)
