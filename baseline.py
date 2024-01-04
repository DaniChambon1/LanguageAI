from feature_hilde import balanced_gen
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

majority_predict = []
for i in range(len(balanced_gen)):
    if balanced_gen['female'][i] == 1:
        majority_predict.append([balanced_gen['Millennial'][balanced_gen['female'] == 1].mode()[0]])
    else:
        majority_predict.append([balanced_gen['Millennial'][balanced_gen['female'] == 0].mode()[0]])


accuracy = accuracy_score(balanced_gen['Millennial'], majority_predict)
precision = precision_score(balanced_gen['Millennial'], majority_predict)
recall = recall_score(balanced_gen['Millennial'], majority_predict)
f1 = f1_score(balanced_gen['Millennial'], majority_predict)
print(f"Majority --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
