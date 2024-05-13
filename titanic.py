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
columns = titanic.columns.tolist()
for col in columns:
    print(f"{col} type -> {titanic[col].dtype}")
print()
# Count missing values in each column
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
print('Results in graphs/')
print()
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
fig, axes = plt.subplots(nrows = 1, ncols = 3)
values_list = [survival_rate, death_rate]
plt.rcParams.update({'font.size': 8})
axes[0].pie(values_list, labels=['Survived', 'Dead'], autopct='%.2f%%')
axes[0].set_title('Survival Rate', fontsize = 12)
values_list = [first_class_percentage, second_class_percentage, third_class_percentage]
axes[1].pie(values_list, labels=['First class', 'Second class', 'Third class'], autopct='%.2f%%')
axes[1].set_title('Class Distribution', fontsize = 12)
values_list = [male_percentage, female_percentage]
axes[2].pie(values_list, labels=['Males', 'Females'], autopct='%.2f%%')
axes[2].set_title('Gender Distribution', fontsize = 12)
plt.tight_layout()
plt.savefig('graphs/task-2-stats.png')

# TASK 3
print('-----TASK 3-----')
print('Results in graphs/')
print()
for col in columns:
    if (titanic[col].dtype in ('int64', 'float64') and col != 'PassengerId'):
        plt.figure()
        titanic[col].hist()
        plt.title(col, fontsize = 12)
        plt.xlabel(col, fontsize = 10)
        plt.ylabel('Number of people', fontsize = 10)
        plt.savefig(f"graphs/{col}.png")

# TASK 4
print('-----TASK 4-----')
print()
for col in columns:
    if titanic[col].dropna().count() != nr_rows:
        missing = nr_rows - titanic[col].dropna().count()
        proportion = missing * 100 / nr_rows
        print(f"{col} missing {proportion:.2f}% values ({missing} out of {nr_rows})")
        temp = titanic[['Survived', col]]
        temp = temp[temp.isnull().any(axis = 1)]
        missing_survived = temp[temp['Survived'] == 1].shape[0]
        proportion = missing_survived * 100 / nr_rows
        print(f"Survived: {proportion:.2f}%")
        missing_dead = temp[temp['Survived'] == 0].shape[0]
        proportion = missing_dead * 100 / nr_rows
        print(f"Died: {proportion:.2f}%")
print()

# TASK 5
print('-----TASK 5-----')
print()
plt.figure()
titanic['Age'].hist(bins = [0, 20, 40, 60, titanic['Age'].max()])
plt.savefig('graphs/task-5-ages.png')
