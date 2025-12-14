import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("train.csv")

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['HasCabin'] = df['Cabin'].notna().astype(int)

df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
df['Title'] = df['Title'].apply(lambda x: x if x in common_titles else 'Other')
df['Title'] = df['Title'].astype('category')

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

bins = [0, 12, 18, 30, 50, 120]
labels = ['Child', 'Teen', 'Adult20_30', 'Adult30_50', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
df['AgeGroup'] = df['AgeGroup'].astype('category')

plt.figure(figsize=(5,4))
sns.countplot(x='Survived', data=df)
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df)
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.show()

plt.figure(figsize=(10,4))
sns.histplot(df['Age'], bins=20)
plt.show()

plt.figure(figsize=(10,4))
sns.kdeplot(data=df, x='Age', hue='Survived', fill=True)
plt.show()

plt.figure(figsize=(10,4))
sns.histplot(df['Fare'], bins=30)
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x='AgeGroup', y='Survived', data=df)
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x='Title', y='Survived', data=df)
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='Embarked', y='Survived', data=df)
plt.show()

numeric = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[numeric].corr(), annot=True, cmap='coolwarm')
plt.show()

df.to_csv("train_cleaned.csv", index=False)
print("Saved cleaned dataset as train_cleaned.csv")