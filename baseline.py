from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
data = pd.read_csv("data/balanced_features.csv")

majority_predict = []
for i in range(len(data)):
    if data['female'][i] == 1:
        majority_predict.append([data['Millennial'][data['female'] == 1].mode()[0]])
    else:
        majority_predict.append([data['Millennial'][data['female'] == 0].mode()[0]])


accuracy = accuracy_score(data['Millennial'], majority_predict)
precision = precision_score(data['Millennial'], majority_predict)
recall = recall_score(data['Millennial'], majority_predict)
f1 = f1_score(data['Millennial'], majority_predict)
print(f"Majority --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
