import pandas as pd
import re
import regex
combined = pd.read_csv("combined_data_english.csv")


def capital(dataset):
    count_list = []
    stand_list = []
    for i in range(len(dataset)):
        count = 0
        for char in dataset['post'][i]:
            if char.isupper():
                count += 1
        count_list.append(count)
        stand_list.append(count / len(dataset['post'][i]))
    return count_list


def emoticons(dataset):
    count_emoticons = []
    for i in range(len(dataset)) :  
        emoticon_pattern = r'(:-?\)|:-?D|;-?\)|:-?P|:-?\(|:-?\/|:-?O|<3)'
        emoticons = re.findall(emoticon_pattern, dataset['post'][i])

        if emoticons:
            count_emoticons.append(1)
        else:
            count_emoticons.append(0)
            
    return count_emoticons

count_emoticons = emoticons(combined)
capital_count = capital(combined)
combined['capital count'] = capital_count
combined['emoticon presence'] = count_emoticons
print(combined['emoticon presence'].corr(combined['birth_year']))
print(combined['capital count'].corr(combined['birth_year']))