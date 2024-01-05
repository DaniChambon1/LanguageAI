from knn2 import kNN
from lr2 import logistic
from svm2 import SVM
from baseline import baseline
from feature_extraction import balanced_gen_features

baseline(balanced_gen_features)
logistic(balanced_gen_features)
SVM(balanced_gen_features)
kNN(balanced_gen_features)