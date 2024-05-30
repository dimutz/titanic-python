"""
Train Titanic Dataset
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### TASK 1

print('-----TASK 1-----')

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

### TASK 2

print('-----TASK 2-----')
print('Results in graphs/')
print()

# Fetch number of survivors and compute percentages
survivors = titanic[titanic['Survived'] == 1].shape[0]
survival_rate = survivors * 100 / nr_rows
death_rate = 100 - survival_rate

# Fetch number of people for each class and compute percentages
first_class = titanic[titanic['Pclass'] == 1].shape[0]
first_class_percentage = first_class * 100 / nr_rows
second_class = titanic[titanic['Pclass'] == 2].shape[0]
second_class_percentage = second_class * 100 / nr_rows
third_class_percentage = 100 - first_class_percentage - second_class_percentage

# Fetch number of males and females and compute percentages
males = titanic[titanic['Sex'] == 'male'].shape[0]
male_percentage = males * 100 / nr_rows
female_percentage = 100 - male_percentage

# Draw pie plots for each percentage set
fig, axes = plt.subplots(nrows = 1, ncols = 3)
values_list = [survival_rate, death_rate]
plt.rcParams.update({'font.size': 8})
axes[0].pie(values_list, labels=['Survived', 'Died'], autopct='%.2f%%')
axes[0].set_title('Survival Rate', fontsize = 12)

values_list = [first_class_percentage, second_class_percentage, third_class_percentage]
axes[1].pie(values_list, labels=['First class', 'Second class', 'Third class'], autopct='%.2f%%')
axes[1].set_title('Class Distribution', fontsize = 12)

values_list = [male_percentage, female_percentage]
axes[2].pie(values_list, labels=['Males', 'Females'], autopct='%.2f%%')
axes[2].set_title('Gender Distribution', fontsize = 12)

plt.tight_layout()
plt.savefig('graphs/task-2-stats.png')
plt.close()

### TASK 3

print('-----TASK 3-----')
print('Results in graphs/')
print()

for col in columns:
    # Find columns with numerical values
    if (titanic[col].dtype in ('int64', 'float64') and col != 'PassengerId'):
        # Plot the values in the found columns
        plt.figure()
        titanic[col].hist()
        plt.title(col, fontsize = 12)
        plt.xlabel(col, fontsize = 10)
        plt.ylabel('Number of people', fontsize = 10)
        plt.savefig(f"graphs/task-3-{col.lower()}.png")
        plt.close()

### TASK 4

print('-----TASK 4-----')

for col in columns:
    # Find columns with missing values
    if titanic[col].dropna().count() != nr_rows:
        # Compute the number of missing values in each found column and percentages
        missing = nr_rows - titanic[col].dropna().count()
        proportion = missing * 100 / nr_rows
        print(f"{col} missing {proportion:.2f}% values ({missing} out of {nr_rows})")
        # Do the same but take into account whether the people survived or not
        temp = titanic[['Survived', col]]
        temp = temp[temp.isnull().any(axis = 1)]
        missing_survived = temp[temp['Survived'] == 1].shape[0]
        proportion = missing_survived * 100 / nr_rows
        print(f"Survived: {proportion:.2f}%")
        missing_dead = temp[temp['Survived'] == 0].shape[0]
        proportion = missing_dead * 100 / nr_rows
        print(f"Died: {proportion:.2f}%")
print()

### TASK 5

print('-----TASK 5-----')

# Set bins and indexes for separating age categories
bins = [0, 20, 40, 60, titanic['Age'].max() + 1]
labels = ['1', '2', '3', '4']

# Add new column
titanic['AgeCat'] = pd.cut(titanic['Age'], bins = bins, labels = labels, right = False)

# Count passagers in each age category
age_counts = titanic['AgeCat'].value_counts().sort_index()
print("Number of people in each age category:")
print(age_counts)

plt.figure()
categories = ['1', '2', '3', '4']
bars = plt.bar(categories, age_counts.values)
# Add numerical value above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
plt.title('Age Categories', fontsize = 12)
plt.xlabel('Age category', fontsize = 10)
plt.ylabel('Number of people', fontsize = 10)
plt.savefig('graphs/task-5-stats.png')
plt.close()
print()

### TASK 6

print('-----TASK 6-----')

# Compute how many men from each age category survived
men = titanic[titanic['Sex'] == 'male']
survived_counts = men[men['Survived'] == 1]['AgeCat'].value_counts().sort_index()
total_counts = men['AgeCat'].value_counts().sort_index()
survival_rates = survived_counts * 100 / total_counts
print("Number of male survivors for each age category:")
print(survived_counts)

plt.figure()
categories = ['1', '2', '3', '4']
bars = plt.bar(categories, survival_rates)
# Add numerical value above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom')
plt.title('Age category influence on survival rates', fontsize = 12)
plt.xlabel('Age category', fontsize = 10)
plt.ylabel('Survival rate', fontsize = 10)
plt.savefig('graphs/task-6-stats.png')
plt.close()
print()

### TASK 7

print('-----TASK 7-----')

# Compute how many children were onboard
children = titanic[titanic['Age'] < 18]
children_proportion = children.shape[0] * 100 / nr_rows
print(f"Percentage of children: {children_proportion:.2f}%")
adults = titanic[titanic['Age'] >= 18]

# Compute survival rates
children_survival_rate = children[children['Survived'] == 1].shape[0] * 100 / children.shape[0]
adults_survival_rate = adults[adults['Survived'] == 1].shape[0] * 100 / adults.shape[0]

plt.figure()
categories = ['Children', 'Adults']
values = [children_survival_rate, adults_survival_rate]
bars = plt.bar(categories, values)
# Add numerical value above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom')
plt.title('Children vs. Adults survival rates', fontsize = 12)
plt.ylabel('Survival rate', fontsize = 10)
plt.savefig('graphs/task-7-stats.png')
plt.close()
print()

### TASK 8

print('-----TASK 8-----')

# Fill missing values with:
# column mean for numerical values
# predominant value for categorical values
for col in columns:
    if nr_rows - titanic[col].dropna().count() != 0:
        if titanic[col].dtype == 'int64':
            titanic[col] = titanic[col].transform(lambda x: x.fillna(int(x.mean())))
        elif titanic[col].dtype == 'float64':
            titanic[col] = titanic[col].transform(lambda x: x.fillna(x.mean()))
        else:
            column_values = titanic[col].value_counts().sort_values(ascending=False)
            predominant_value = column_values.keys()[0].split()[0]
            titanic[col] = titanic[col].transform(lambda x: x.fillna(predominant_value))
# Save modified dataset
titanic.to_csv('data/new_train.csv')
print("Done")
print()

### TASK 9

print('-----TASK 9-----')

def extract_title(name):
    """
       Function for extracting the title of a person
       name(str) - a person's name
       Returns (str) - title of a person or null string 
    """
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# Create new column for titles
titanic['Title'] = titanic['Name'].apply(extract_title)
# Check the title - sex correspondence
sex_check = titanic.groupby(['Title', 'Sex']).size().unstack(fill_value = 0)
print(sex_check)

plt.figure(figsize = (12,6))
bars = plt.bar(titanic['Title'].value_counts().index, titanic['Title'].value_counts().values)
# Add numerical value above each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, yval, ha='center', va='bottom')
plt.title('Title count', fontsize = 12)
plt.ylabel('Number of people', fontsize = 10)
plt.xticks(rotation = 45)
plt.savefig('graphs/task-9-stats.png')
plt.close()
print()

### TASK 10

print('-----TASK 10-----')

print('Results in graphs/')
print()

# Create new column for being alone onboard (no siblings, spouses, parents or children onboard)
titanic['isAlone'] = (titanic['SibSp'] + titanic['Parch'] == 0).astype(int)

plt.figure()
ax = sns.histplot(data = titanic, x = 'isAlone', hue = 'Survived', multiple = 'dodge', bins = 2)
# Add numerical value above each bar
for bar in ax.containers:
    ax.bar_label(bar, label_type = 'edge')
plt.title('Being alone influence on survival')
plt.ylabel('Number of people')
plt.xticks([0, 1], ['Not alone', 'Alone'])
plt.savefig('graphs/task-10-stats-1.png')
plt.close()

# Plot relation between fare, class and survival for the first 100 data entries
plt.figure(figsize = (16, 8))
sns.catplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = titanic.head(100), kind = 'swarm',
            height = 6, aspect = 2)
plt.title('Relation between fare, class and survival (first 100 entries)', fontsize = 12)
plt.xlabel('Pclass', fontsize = 10)
plt.ylabel('Fare', fontsize = 10)
plt.tight_layout()
plt.savefig('graphs/task-10-stats-2.png')
plt.close()
