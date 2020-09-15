# %%
# imports
import pandas as pd

# %%
# create all dataframes

csv_names = ['CPI_3.8_Abuse_Neglect_Investigations_-_Victims_with_Demographics_by_Region_FY2010-FY2019', 
            'CPS_2.1_Removals_-_by_Region_with_Child_Demographics_FY2010-2019', 
            'CPS_3.1_Placement_Types_of_Children_in_Substitute_Care_During_the_Fiscal_Year_by_County_with_Demographics_FY2010-2019']
            # 'CPS_3.2_in_Substitute_Care_on_August_31_by_Placement_Type_with_Demographics_FY2010-2019', 
            # 'CPS_4.1_Adoption_-_Children_Waiting_For_Adoption_on_31_August_by_Region_with_Demographics_FY2010-2019', 
            # 'CPS_4.3_Adoption_-_Children_In_Adoption_Placements_by_Region_with_Demographics_FY2010-2019', 
            # 'CPS_4.4_Adoption_-_Disabling_Conditions_of_Children_in_Adoption_Placements_by_Region_with_Demographics_FY2010-2019', 
            # 'CPS_4.5_Adoptions_Consummated_by_Region_with_Demographics_FY2010-2019']

dfs = {}
for i, csv in enumerate(csv_names):
    dfs[i] = pd.read_csv(f'data/{csv}.csv')
# %%

# COME UP WITH A BETTER WAY TO DO THIS

df0 = dfs[0]
df1 = dfs[1]
df2 = dfs[2]
# df3 = dfs[3]
# df4 = dfs[4]
# df5 = dfs[5]
# df6 = dfs[6]
# df7 = dfs[7]
# %%
# final_df = df2.copy()
# %%
final_df.info()
# %%
final_df.drop(['County'], axis=1, inplace=True)
# %%
# bucketize the 'Age' column into 'Age Group'

bins = [0, 5, 12, 17]
labels = ['Birth to Five Years Old', 'Six to Twelve Years Old', 'Thirteen to Seventeen Years Old']
final_df['Age Group'] = pd.cut(final_df['Age'], bins, labels=labels)
final_df.drop(['Age'], axis=1, inplace=True)

# %%
type_subs_care_count = final_df.groupby(['Fiscal Year','Region', 'Gender', 'Race/Ethnicity', 'Age Group', 'Type of Substitute Care']).count()['Placed with Relative']
# pd.pivot_table(type_subs_care_count, values=type_subs_care_count['Type of Substitute Care'], index = ['Fiscal Year', 'Region', 'Gender', 'Race/Ethnicity', 'Age Group'])
df_new = pd.pivot_table(type_subs_care_count, values='Type of Substitute Care', index = ['Fiscal Year', 'Region', 'Gender', 'Race/Ethnicity', 'Age Group'])
df_new.head()
# %%
pd.pivot_table(final_df, values='Placed with Relative', index = ['Fiscal Year', 'Region', 'Gender', 'Race/Ethnicity', 'Age Group'], columns=['Type of Substitute Care'], aggfunc='count')
# %%
type_subs_care_count.reset_index()
# %%
type(type_subs_care_count)# %%

final_df.groupby(['Fiscal Year','Region', 'Gender', 'Race/Ethnicity', 'Age Group', 'Type of Substitute Care']).size().join(final_df, on=final_df['Fiscal Year', 'Region', 'Gender', 'Race/Ethnicity', 'Age Group'], how='left')
# %%
pd.merge(final_df, type_subs_care_count, left_on=['Fiscal Year','Region', 'Gender', 'Race/Ethnicity', 'Age Group', 'Type of Substitute Care'], right_index=True)
# %
# %%
final_df.join(type_subs_care_count, on=final_df['Fiscal Year', 'Region', 'Gender', 'Race/Ethnicity', 'Age Group'], how='left')
type_subs_care_count.merge(final_df, how='left')
# %%
final_df.crosstab['Fiscal Year','Region', 'Gender', 'Race/Ethnicity', 'Age Group', 'Type of Substitute Care']
# %%
final_df.head()

# %%
final_df.info()
# %%
final_df['Demographic Combination'] = final_df['Fiscal Year'] + final_df['Region'] + final_df['Gender'] + final_df['Race/Ethnicity'] + final_df['Age Group']
# %%
final_df
# %%
