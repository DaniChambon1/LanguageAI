# LanguageAI
Language and AI assignment

It all starts with gender_age_combined.csv. This file was created by running the following SQL statement on the data that was provided to us.

SELECT DISTINCT birth_year.post, gender.auhtor_ID, gender.female, birth_year.birth_year
FROM gender, birth_year
WHERE gender.auhtor_ID == birth_year.auhtor_ID


