import pandas
import re
from data_cleaning import combined_gen as df

def contraction(dataset):
    contraction = '[a-zA-Z]+\'m\s|[a-zA-Z]+\'d\s|[a-zA-Z]+\'ll\s|[a-zA-Z]+\'re\s|[a-zA-Z]+\'ve\s|[a-zA-Z]+\'s\s|[a-zA-Z]+\'t\s'
    lst_contractions = []
    for i in range(len(dataset)):
        post = dataset['post'][i]
        find_contraction = re.findall(contraction, post)
        total_contractions_sentence = len(find_contraction)
        lst_contractions.append(total_contractions_sentence)
    return lst_contractions


def exaggerate(dataset):
    exaggeration = '[a|A]{3,}|[b|B]{3,}|[c|C]{3,}|[d|D]{3,}|[e|E]{3,}|[f|F]{3,}|[g|G]{3,}|[h|H]{3,}|[i|I]{3,}|[j|J]{3,}|[k|K]{3,}|[l|L]{3,}|[m|M]{3,}[n|N]{3,}|[o|O]{3,}|[p|P]{3,}|[q|Q]{3,}|[r|R]{3,}|[s|S]{3,}|[t|T]{3,}|[u|U]{3,}|[v|V]{3,}|[w|W]{3,}|[x|X]{3,}|[y|Y]{3,}|[z|Z]{3,}|[.]{2,}|[!]{2,}'
    lst_exaggeration = []
    for i in range(len(dataset)):
        post = dataset['post'][i]
        find_exaggeration = re.findall(exaggeration, post)
        total_exaggeration_sentence = len(find_exaggeration)
        lst_exaggeration.append(total_exaggeration_sentence)
    return lst_exaggeration


df['contraction'] = contraction(df)
df['exaggerate'] = exaggerate(df)