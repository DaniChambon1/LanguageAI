from kNN import kNN
from LR import logistic
from SVM import SVM
from baseline import baseline
#from feature_extraction import balanced_gen_features
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

balanced_gen_features = pd.read_csv("data/balanced_features.csv", index_col=[0])
baseline(balanced_gen_features)
y_test_lr, y_pred_lr = logistic(balanced_gen_features)
y_test_svm, y_pred_svm = SVM(balanced_gen_features)
y_test_knn, y_pred_knn = kNN(balanced_gen_features)

plt.figure(figsize=(8, 6))

# Train and plot ROC curves for each model

fpr, tpr, _ = roc_curve(y_test_lr, y_pred_lr)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'{"LR"} (AUC = {roc_auc:.2f})')
fpr, tpr, _ = roc_curve(y_test_svm, y_pred_svm)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'{"SVM"} (AUC = {roc_auc:.2f})')
fpr, tpr, _ = roc_curve(y_test_knn, y_pred_knn )
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'{"kNN"} (AUC = {roc_auc:.2f})')

# Plotting settings
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
