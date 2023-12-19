from feature_hilde import combined_gen
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

majority_predict = []
for i in range(len(combined_gen)):
    if combined_gen['female'][i] == 1:
        majority_predict.append([combined_gen['Millennial'][combined_gen['female'] == 1].mode()[0]])
    else:
        majority_predict.append([combined_gen['Millennial'][combined_gen['female'] == 0].mode()[0]])


accuracy = accuracy_score(combined_gen['Millennial'], majority_predict)
precision = precision_score(combined_gen['Millennial'], majority_predict)
recall = recall_score(combined_gen['Millennial'], majority_predict)
f1 = f1_score(combined_gen['Millennial'], majority_predict)
print(f"Majority --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")