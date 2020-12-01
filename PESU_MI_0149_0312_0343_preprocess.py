#importing the libraries
import pandas as pd

#Function to fill the missing data with the mean of the series
def impute_mean(series):
    return series.fillna(series.mean())

#Function to fill the missing data with the median of the series
def impute_median(series):
    return series.fillna(series.median())

#Loading the dataset
df = pd.read_csv("LBW_Dataset.csv")

#Grouping the dataset with the Community column
df2 = df.groupby(['Community'])

#On the basis of community column the missing values are calculated and filled
df.Education = df2['Education'].transform(impute_median)
df.Age = df2['Age'].transform(impute_mean)
df.Weight = df2['Weight'].transform(impute_mean)
df.BP = df2['BP'].transform(impute_mean)
df.HB = df2['HB'].transform(impute_mean)
df['Delivery phase'] = df2['Delivery phase'].transform(impute_median)
df['Residence'] = df2['Residence'].transform(impute_median)

"""
df2 = pd.DataFrame(df, columns=['Community', 'Age'])
df2.groupby('Community').mean().reset_index()
"""
#Exporting the dataframe to csv file after cleaning the dataset
df.to_csv(r'cleaned_LBW_Dataset.csv', index = False)
