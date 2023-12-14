import pandas as pd
import re
from data_cleaning import combined_gen

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
    return count_list, stand_list


def emoticons(dataset):
    count_emoticons = []
    for i in range(len(dataset)) :  
        emoticon_pattern = r'(:-?\))'
        emoticons = re.findall(emoticon_pattern, dataset['post'][i])
        if emoticons:
            count_emoticons.append(1)
        else:
            count_emoticons.append(0)
            
    return count_emoticons



combined_gen['capital count'], combined_gen['capital count (stand.)'] =  capital(combined_gen)
combined_gen['emoticon presence'] = emoticons(combined_gen)
