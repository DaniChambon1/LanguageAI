from feature_hilde import combined_gen
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

majority_predict = [combined_gen['Millennial'].mode()[0]]*len(combined_gen)

accuracy = accuracy_score(combined_gen['Millennial'], majority_predict)
precision = precision_score(combined_gen['Millennial'], majority_predict)
recall = recall_score(combined_gen['Millennial'], majority_predict)
f1 = f1_score(combined_gen['Millennial'], majority_predict)
print(f"Majority --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")