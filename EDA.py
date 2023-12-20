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
    print("There are no authors who are classified as male and female.")
else:
    print("PROBLEM DETECTED: THERE EXIST AUTHORS WHO ARE CLASSIFIED AS MALE AND FEMALE")

nr_female_authors = sum(grouped2['female', 'min'])
nr_male_authors = len(grouped2) - nr_female_authors
print(f"The total number of authors is {len(grouped2)}. The number of male authors is {nr_male_authors} and the number of female authors is {nr_female_authors}.")


### Get the number of millenials and gen z
nr_millennials = len(combined_gen[combined_gen["Millennial"] == 1])
nr_genz = len(combined_gen[combined_gen["Millennial"] == 0])
print(f"The number of millenials is {nr_millennials}. The number of persons classified as gen z is {nr_genz}.")


### Get the number of millenials and gen z posts that are male vs female
nr_female_millennials = len(combined_gen[(combined_gen["female"] == 1) & (combined_gen["Millennial"] == 1)])
nr_female_genz = len(combined_gen[(combined_gen["female"] == 1) & (combined_gen["Millennial"] == 0)])
print(f"The number of female millenials is {nr_female_millennials}. The number of females classified as gen z is {nr_female_genz}.")

nr_male_millennials = len(combined_gen[(combined_gen["female"] == 0) & (combined_gen["Millennial"] == 1)])
nr_male_genz = len(combined_gen[(combined_gen["female"] == 0) & (combined_gen["Millennial"] == 0)])
print(f"The number of male millenials is {nr_male_millennials}. The number of males classified as gen z is {nr_male_genz}")

