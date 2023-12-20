import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from data_cleaning import combined_gen

### Get the number of posts made by males and females 
grouped_data = combined_gen.groupby('female').size()

count_male = grouped_data[0] if 0 in grouped_data.index else 0
count_female = grouped_data[1] if 1 in grouped_data.index else 0

print(f"The total number of posts is {len(combined_gen)}.The number of posts made by males is {count_male} and the number of posts made by females is {count_female}.")


### Get the number of authors that are male and female
grouped2 = combined_gen.groupby('auhtor_ID').agg({'female': ['min', 'max']})

# Check if there are no persons who are classified as both female and male
no_double_classifications = grouped2['female', 'min'].equals(grouped2['female', 'max'])
if no_double_classifications == True:
    print("There are not authors who are classified as male and female.")
else:
    print("PROBLEM DETECTED: THERE EXIST AUTHORS WHO ARE CLASSIFIED AS MALE AND FEMALE")

nr_female_authors = sum(grouped2['female', 'min'])
nr_male_authors = len(grouped2) - nr_female_authors
print(f"The total number of authors is {len(grouped2)}. The number of male authors is {nr_male_authors} and the number of female authors is {nr_female_authors}.")


