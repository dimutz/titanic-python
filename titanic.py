"""
Train Titanic Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt

### TASK 1
print('-----TASK 1-----')
print()
# Open .csv file
titanic = pd.read_csv('data/train.csv')
# Fetch DataFrame sizes
(nr_rows, nr_cols) = titanic.shape
print(f'Number of lines and columns: {nr_rows}, {nr_cols}')
print()
# Show data types for each column
print(titanic.dtypes)
print()
# Count missing values in each column
columns = titanic.columns.tolist()
for col in columns:
    full_rows = titanic[col].dropna().count()
    print(f"Column '{col}' missing {nr_rows - full_rows} values")
print()
# Check for duplicate lines
duped = titanic.loc[titanic.index.duplicated(), :]
print(f"Duplicate lines? {duped.shape[0] != 0}")
print()

# TASK 2
print('-----TASK 2-----')
print()
print('Results in graphs/')
survivors = titanic[titanic['Survived'] == 1].shape[0]
survival_rate = survivors * 100 / nr_rows
death_rate = 100 - survival_rate
first_class = titanic[titanic['Pclass'] == 1].shape[0]
first_class_percentage = first_class * 100 / nr_rows
second_class = titanic[titanic['Pclass'] == 2].shape[0]
second_class_percentage = second_class * 100 / nr_rows
third_class = titanic[titanic['Pclass'] == 3].shape[0]
third_class_percentage = third_class * 100 / nr_rows
males = titanic[titanic['Sex'] == 'male'].shape[0]
male_percentage = males * 100 / nr_rows
female_percentage = 100 - male_percentage
plt.pie([survival_rate, death_rate], labels=['Survivors', 'Dead'], autopct='%.2f%%')
plt.savefig('graphs/survival_rate.png')
plt.close()
class_list = [first_class_percentage, second_class_percentage, third_class_percentage]
plt.pie(class_list, labels=['First class', 'Second class', 'Third class'], autopct='%.2f%%')
plt.savefig('graphs/class_distribution.png')
plt.close()
plt.pie([male_percentage, female_percentage], labels=['Males', 'Females'], autopct='%.2f%%')
plt.savefig('graphs/gender_distribution.png')
plt.close()
