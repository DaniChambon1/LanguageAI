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

# Now, majority prediction will be done without gender as a predictor

majority_predict2 = [1] * len(combined_gen)

accuracy2 = accuracy_score(combined_gen['Millennial'], majority_predict2)
precision2 = precision_score(combined_gen['Millennial'], majority_predict2)
recall2 = recall_score(combined_gen['Millennial'], majority_predict2)
f12 = f1_score(combined_gen['Millennial'], majority_predict2)
print(f"Majority predict without gender --> Accuracy: {accuracy2}, Precision: {precision2}, Recall: {recall2}, F1-score: {f12}")
