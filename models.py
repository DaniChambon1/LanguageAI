from kNN import kNN
from LR import logistic
from SVM import SVM
from baseline import baseline
from feature_extraction import balanced_gen_features

baseline(balanced_gen_features)
logistic(balanced_gen_features)
SVM(balanced_gen_features)
kNN(balanced_gen_features)