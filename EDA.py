import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from data_cleaning import combined_gen


### Get the number of posts made by males and females 
count_female = len(combined_gen[combined_gen["female"] == 0])
count_male = len(combined_gen[combined_gen["female"] == 0])
print(f"The total number of posts is {len(combined_gen)}. The number of posts made by males is {count_male} and the number of posts made by females is {count_female}.")


### Get the number of authors that are male and female
grouped2 = combined_gen.groupby('auhtor_ID').agg({'female': ['min', 'max'], 'Millennial': ['min']})

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
print(f"The number of female millenials posts is {nr_female_millennials}. The number of female posts classified as gen z is {nr_female_genz}.")

nr_male_millennials = len(combined_gen[(combined_gen["female"] == 0) & (combined_gen["Millennial"] == 1)])
nr_male_genz = len(combined_gen[(combined_gen["female"] == 0) & (combined_gen["Millennial"] == 0)])
print(f"The number of male millenials posts is {nr_male_millennials}. The number of male posts classified as gen z is {nr_male_genz}")


### Get the number of authors that are millenial and gen z for males and females

nr_female_authors_millennial = len(grouped2[(grouped2['female']['min'] == 1) & (grouped2['Millennial']['min'] == 1)])
nr_female_authors_genz = len(grouped2[(grouped2['female']['min'] == 1) & (grouped2['Millennial']['min'] == 0)])
print(f"The number of authors that are female and millennial is {nr_female_authors_millennial}. The number of authors that are female and gen z is {nr_female_authors_genz}.")

nr_male_authors_millennial = len(grouped2[(grouped2['female']['min'] == 0) & (grouped2['Millennial']['min'] == 1)])
nr_male_authors_genz = len(grouped2[(grouped2['female']['min'] == 0) & (grouped2['Millennial']['min'] == 0)])
print(f"The number of authors that are male and millennial is {nr_male_authors_millennial}. The number of authors that are male and gen z is {nr_male_authors_genz}.")



### Make a graph of the distribution of ages in the dataset.

#combined_gen["birth_year"].plot(kind='bar')
# grouped_by = combined_gen.groupby('birth_year').size()
# grouped_by.loc[2009] = 0
# grouped_by = grouped_by.sort_index()
# grouped_by.plot(kind='bar')
# plt.xlabel('Birth Year')
# plt.ylabel('Count')
# plt.title('Count of Birth Years of posts', size = 16)
# plt.show()


grouped_by_gender = combined_gen.groupby(['birth_year', 'female']).size().unstack(fill_value=0)
# Adding missing year (2009) with count 0 for both genders
grouped_by_gender.loc[2009] = [0, 0]
grouped_by_gender = grouped_by_gender.sort_index()
ax = grouped_by_gender.plot(kind='bar', stacked=True)
plt.xlabel('Birth Year')
plt.ylabel('Count')
plt.title('Count of Birth Years by Gender', size=16)
plt.legend(['Male', 'Female'], loc='upper right')
# Line is not shown??
ax.axvline(x=1996, color='red', linestyle='--')
plt.show()