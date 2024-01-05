from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def baseline(data):
    # Determine majority
    majority_predict = []
    for i in range(len(data)):
        # if author of post is female, predict majority generation of female authors
        if data['female'][i] == 1:
            majority_predict.append([data['Millennial'][data['female'] == 1].mode()[0]])
        # if author of post is male, predict majority generation of male authors
        else:
            majority_predict.append([data['Millennial'][data['female'] == 0].mode()[0]])

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(data['Millennial'], majority_predict)
    precision = precision_score(data['Millennial'], majority_predict)
    recall = recall_score(data['Millennial'], majority_predict)
    f1 = f1_score(data['Millennial'], majority_predict)
    print(f"Majority --> Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
