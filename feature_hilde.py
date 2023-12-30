import pandas as pd
import re
from tqdm import tqdm
from langdetect import detect
import string
from data_cleaning import combined_gen

def capital(dataset):
    stand_capitals = []
    for i in range(len(dataset)):
        count = 0
        for char in dataset['post'][i]:
            if char.isupper():
                count += 1
        stand_capitals.append(count / len(dataset['post'][i]))
    return stand_capitals


def emoticons(dataset):
    stand_emoticons = []
    for i in range(len(dataset)) :  
        emoticon_pattern = r'(:-?\)|:-?D|;-?\)|:-?P|:-?\(|:-?\/|:-?O|<3)'
        emoticons = re.findall(emoticon_pattern, dataset['post'][i])
        stand_emoticons.append(len(emoticons) / len(dataset['post'][i]))
    return stand_emoticons

def pronouns(dataset):
    stand_pronouns = []
    for i in range(len(dataset)) :  
        pronoun_pattern = r'\b(?:he|she|it|they|we|you|I|me|my|mine|you|your|yours|him|her|hers|us|our|ours|them|their|theirs)\b'
        pronouns = re.findall(pronoun_pattern, dataset['post'][i])
        stand_pronouns.append(len(pronouns) / len(dataset["post"][i].split()))
    return stand_pronouns


def contraction(dataset):
    contraction = '[a-zA-Z]+\'m\s|[a-zA-Z]+\'d\s|[a-zA-Z]+\'ll\s|[a-zA-Z]+\'re\s|[a-zA-Z]+\'ve\s|[a-zA-Z]+\'s\s|[a-zA-Z]+\'t\s'
    lst_contractions = []
    for i in range(len(dataset)):
        post = dataset['post'][i]
        find_contraction = re.findall(contraction, post)
        total_contractions_sentence = len(find_contraction)
        lst_contractions.append(total_contractions_sentence / len(dataset["post"][i].split()))
    return lst_contractions


def exaggerate(dataset):
    exaggeration = '[a|A]{3,}|[b|B]{3,}|[c|C]{3,}|[d|D]{3,}|[e|E]{3,}|[f|F]{3,}|[g|G]{3,}|[h|H]{3,}|[i|I]{3,}|[j|J]{3,}|[k|K]{3,}|[l|L]{3,}|[m|M]{3,}[n|N]{3,}|[o|O]{3,}|[p|P]{3,}|[q|Q]{3,}|[r|R]{3,}|[s|S]{3,}|[t|T]{3,}|[u|U]{3,}|[v|V]{3,}|[w|W]{3,}|[x|X]{3,}|[y|Y]{3,}|[z|Z]{3,}|[.]{2,}|[!]{2,}'
    lst_exaggeration = []
    for i in range(len(dataset)):
        post = dataset['post'][i]
        find_exaggeration = re.findall(exaggeration, post)
        total_exaggeration_sentence = len(find_exaggeration)
        lst_exaggeration.append(total_exaggeration_sentence / len(dataset["post"][i].split()))
    return lst_exaggeration


def punctuation(dataset):
    post_list = list(dataset["post"])
    
    punctuation_count = 0
    punctuation_count_standardized_list = []

    comma_count = 0
    comma_count_standardized_list = []

    exclamation_count = 0
    exclamation_count_standardized_list = []

    for post in tqdm(post_list):
        for character in post:
            if character in string.punctuation:
                punctuation_count += 1
            if character == ",":
                comma_count += 1
            if character == "!":
                exclamation_count += 1
        
        punctuation_count_standardized_list.append(punctuation_count / len(post))
        punctuation_count = 0

        comma_count_standardized_list.append(comma_count / len(post))
        comma_count = 0

        exclamation_count_standardized_list.append(exclamation_count /len(post))
        exclamation_count = 0
    
    return punctuation_count_standardized_list, comma_count_standardized_list, exclamation_count_standardized_list

combined_gen['contraction count'] = contraction(combined_gen)
combined_gen['exaggeration count'] = exaggerate(combined_gen)
combined_gen['capital count'] =  capital(combined_gen)
combined_gen['emoticon count'] = emoticons(combined_gen)
combined_gen['pronoun count'] = pronouns(combined_gen)
combined_gen['punctuation count'], combined_gen['comma count'], combined_gen['exclamation count'] = punctuation(combined_gen)


combined_gen.to_csv("data\combined_gen.csv")