import pandas as pd
import numpy as np
# birth_year = pd.read_csv("lai-data/birth_year.csv")
# gender = pd.read_csv("lai-data/gender.csv")
combined = pd.read_csv("combined_data_english.csv")

generations = []
for i in range(len(combined)):
    if combined['birth_year'][i] in range(1980, 1997):
        generations.append(1)
    elif combined['birth_year'][i] in range(1997,2013):
        generations.append(0)
    else:
        generations.append(-1)

combined_gen = combined.copy()
combined_gen['Millennial'] = generations
combined_gen = combined_gen[combined_gen['Millennial']!=-1].reset_index(drop=True)