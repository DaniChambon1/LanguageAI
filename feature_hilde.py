import pandas as pd
import re
from data_cleaning import combined_gen

def capital(dataset):
    count_capitals = []
    stand_capitals = []
    for i in range(len(dataset)):
        count = 0
        for char in dataset['post'][i]:
            if char.isupper():
                count += 1
        count_capitals.append(count)
        stand_capitals.append(count / len(dataset['post'][i]))
    return count_capitals, stand_capitals


def emoticons(dataset):
    count_emoticons = []
    stand_emoticons = []
    for i in range(len(dataset)) :  
        emoticon_pattern = r'(:-?\)|:-?D|;-?\)|:-?P|:-?\(|:-?\/|:-?O|<3)'
        emoticons = re.findall(emoticon_pattern, dataset['post'][i])
        count_emoticons.append(len(emoticons))
        stand_emoticons.append(len(emoticons) / len(dataset['post'][i]))
    return count_emoticons, stand_emoticons

def pronouns(dataset):
    count_pronouns = []
    stand_pronouns = []
    for i in range(len(dataset)) :  
        pronoun_pattern = r'\b(?:he|she|it|they|we|you|I|me|my|mine|you|your|yours|him|her|hers|us|our|ours|them|their|theirs)\b'
        pronouns = re.findall(pronoun_pattern, dataset['post'][i])
        count_pronouns.append(len(pronouns))
        stand_pronouns.append(len(pronouns) / len(dataset["post"][i].split()))
    return count_pronouns, stand_pronouns



combined_gen['capital count'], combined_gen['capital count (stand.)'] =  capital(combined_gen)
combined_gen['emoticon count'], combined_gen['emoticon count (stand.)'] = emoticons(combined_gen)
combined_gen['pronoun count'], combined_gen['pronoun count (stand.)'] = pronouns(combined_gen)
