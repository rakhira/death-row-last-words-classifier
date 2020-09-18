# %%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('data/CPS_df.csv')
# %%
df.drop(['Unnamed: 0'], axis=1, inplace=True)
# %%
df.describe()
# %%
df.info()
# %%
df['Number of Consummated Adoptions'] = df['Same Race/Ethnicity as Adoptive Parent'] + df['Different Race/Ethnicity as Adoptive Parent']
# %%
df['Number of Kids in Substitute Care'] = df['In Foster Care'] + df['In Other Substitute Care']
# %%
df['Percent of Kids in Consummated Adoptions'] = 100*((df['Same Race/Ethnicity as Adoptive Parent'] + df['Different Race/Ethnicity as Adoptive Parent'])/(df['Number of Kids in Substitute Care']+df['Number of Consummated Adoptions']))
# %%
df.info()
# %%
df.fillna(0, inplace=True)
# %%
df_0 = df[df['Percent of Kids in Consummated Adoptions']==0]
# %%
df_0.head()
# %%
df_0.info()
# %%
df_0['Region'].value_counts()
# %%
df_0['Number of Kids in Substitute Care'].value_counts(ascending=False).mean()
# %%
# majority of 0s are for Asian and Other, 10-15% for other races
df_0['Race/Ethnicity'].value_counts()
# %%
df['Race/Ethnicity'].value_counts()
# %%
df_0['Age Group'].value_counts()
# %%
model_df = pd.get_dummies(df, columns=['Fiscal Year', 'Region', 'Race/Ethnicity', 'Age Group', 'Gender'])
# %%
model_df.head()
# %%
model_df.info()
# %%
model_df.to_csv('data/model_df.csv')
# %%
# %%
sns.distplot(df['Percent of Kids in Consummated Adoptions']).set_title('Distribution of Percent of Kids in Consummated Adoptions')
# %%
sns.boxplot([df['Average Months Since Termination of Parental Rights'], df['Percent of Kids in Consummated Adoptions']]).set_title('Average Months Since Termination of Parental Rights vs. Percent of Kids in Consummated Adoptions')
# %%
sns.catplot(x='Gender', y='Percent of Kids in Consummated Adoptions', data=df)
# %%
sns.catplot(x='Gender', y='Same Race/Ethnicity as Adoptive Parent', data=df)
# %%
sns.catplot(x='Race/Ethnicity', y='Same Race/Ethnicity as Adoptive Parent', data=df)
# %%
sns.catplot(x="Race/Ethnicity", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="swarm", data=df)
# %%
# fig, ax = plt.subplots(figsize = (15, 20))
# ax = sns.catplot(x="Region", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="swarm", data=df, figsize = ())
# %%
sns.catplot(x="Race/Ethnicity", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="box", data=df)
# %%
sns.catplot(x="Region", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="box", data=df)
# %%
sns.catplot(x="Region", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="boxen", data=df)
# %%
sns.catplot(x="Race/Ethnicity", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="boxen", data=df)
# %%
sns.catplot(x="Race/Ethnicity", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="bar", data=df)
# %%
sns.catplot(x="Race/Ethnicity", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="box", data=df)
# %%
sns.catplot(x="Race/Ethnicity", y="Percent of Kids in Consummated Adoptions", hue="Gender", kind="bar", data=df)
# %%
df.head()
# %%
# do a count by race of total kids in system
sns.catplot(x='Gender', y='Percent of Kids in Consummated Adoptions', hue="Age Group", kind="box", data=df[df['Race/Ethnicity'] == 'African American'])
# %%
# df[df['Race/Ethnicity'] == 'African American'].nunique(df['Age Group'])
# %%
# do a scatter matrix on continuous variables
sns.pairplot(df, hue='Gender')
# %%
df.value_counts()
# %%
df[df['Race/Ethnicity'] == 'African American']['Age Group'].value_counts()
# df[df['Race/Ethnicity'] == 'African American']['Region'].value_counts(ascending=True)
# df[df['Race/Ethnicity'] == 'African American']['Gender'].value_counts()
# df[df['Race/Ethnicity'] == 'African American']['Fiscal Year'].value_counts()
# %%
df[df['Race/Ethnicity'] == 'Anglo']['Age Group'].value_counts()
df[df['Race/Ethnicity'] == 'Anglo']['Region'].value_counts(ascending=True)
df[df['Race/Ethnicity'] == 'Anglo']['Gender'].value_counts()
df[df['Race/Ethnicity'] == 'Anglo']['Fiscal Year'].value_counts()
# %%
df[df['Race/Ethnicity'] == 'Hispanic']['Age Group'].value_counts()
df[df['Race/Ethnicity'] == 'Hispanic']['Region'].value_counts(ascending=True)
df[df['Race/Ethnicity'] == 'Hispanic']['Gender'].value_counts()
df[df['Race/Ethnicity'] == 'Hispanic']['Fiscal Year'].value_counts()
# %%
df[df['Race/Ethnicity'] == 'Asian']['Age Group'].value_counts()
df[df['Race/Ethnicity'] == 'Asian']['Region'].value_counts(ascending=True)
df[df['Race/Ethnicity'] == 'Asian']['Gender'].value_counts()
df[df['Race/Ethnicity'] == 'Asian']['Fiscal Year'].value_counts()
# %%
df[df['Race/Ethnicity'] == 'Other']['Age Group'].value_counts()
df[df['Race/Ethnicity'] == 'Other']['Region'].value_counts(ascending=True)
df[df['Race/Ethnicity'] == 'Other']['Gender'].value_counts()
df[df['Race/Ethnicity'] == 'Other']['Fiscal Year'].value_counts()
# %%
df[(df['Race/Ethnicity'] == 'African American') & (df['Age Group'] == 'Birth to Five Years Old')]. sum()['Number of Consummated Adoptions']
df[(df['Race/Ethnicity'] == 'African American') & (df['Age Group'] == 'Six to Twelve Years Old')]. sum()['Number of Consummated Adoptions']
# %%
df[(df['Race/Ethnicity'] == 'Anglo') & (df['Age Group'] == 'Birth to Five Years Old')]. sum()['Number of Consummated Adoptions']
df[(df['Race/Ethnicity'] == 'Anglo') & (df['Age Group'] == 'Six to Twelve Years Old')]. sum()['Number of Consummated Adoptions']
# %%
df[(df['Race/Ethnicity'] == 'Hispanic') & (df['Age Group'] == 'Birth to Five Years Old')]. sum()['Number of Consummated Adoptions']
df[(df['Race/Ethnicity'] == 'Hispanic') & (df['Age Group'] == 'Six to Twelve Years Old')]. sum()['Number of Consummated Adoptions']
# %%
df[(df['Race/Ethnicity'] == 'Asian') & (df['Age Group'] == 'Birth to Five Years Old')]. sum()['Number of Consummated Adoptions']
df[(df['Race/Ethnicity'] == 'Asian') & (df['Age Group'] == 'Six to Twelve Years Old')]. sum()['Number of Consummated Adoptions']
# %%
df[(df['Race/Ethnicity'] == 'Other') & (df['Age Group'] == 'Birth to Five Years Old')]. sum()['Number of Consummated Adoptions']
df[(df['Race/Ethnicity'] == 'Other') & (df['Age Group'] == 'Six to Twelve Years Old')]. sum()['Number of Consummated Adoptions']
# %%
df[(df['Race/Ethnicity'] == 'African American') & (df['Age Group'] == 'Birth to Five Years Old')]. sum()['Number of Kids in Substitute Care']
# df[(df['Race/Ethnicity'] == 'African American') & (df['Age Group'] == 'Six to Twelve Years Old')]. sum()['Number of Kids in Substitute Care']
# %%
sns.pairplot(df, hue='Gender')
# %%
df.fillna(0, inplace=True)