import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


def main():
    sns.set()
    raw_data=pd.read_csv('1.04. Real-life example.csv')
    pd.set_option('display.expand_frame_repr', False)
    print(raw_data.describe(include='all'))
        ## it will be hard to implement model that has 312 unique models so will drop it
    data=raw_data.drop(['Model'],axis=1)
    #print(data.describe(include='all'))
    #print(data.isnull().sum())
        #price and Enginev has null so drop the null entrys
    data_no_missing_values=data.dropna(axis=0) #drop all the entrys with missing values
    #print(data_no_missing_values.describe(include='all'))
    #sns.distplot(data_no_missing_values['Price'])
    #plt.show()
         # we can see we have strong outliners in the graph and in the describe the max value is 300000 far from the mean 19552
         # we can drop the 0.99% of high prices to get rid of the outliners
    q=data_no_missing_values['Price'].quantile(0.99)
    data_1=data_no_missing_values[data_no_missing_values['Price']<q]
         # getting all the entrys that are less than 99% of the max price
    #print(data_1.describe(include='all'))
         # the max is closer to the mean now lets plot the data
    #sns.distplot(data_1['Price'])
    #plt.show()
         #less outliners
         #deal with the mileage the same way
    q=data_1['Mileage'].quantile(0.99)
    data_2=data_1[data_1['Mileage']<q]
    #sns.distplot(data_2['Mileage'])
    #plt.show()
    data_3=data_2[data_2['EngineV']<6.5] #EngineV cant be above 6.5 from google
         #year has outliners in the low precent so get rid of them
    q=data_3['Year'].quantile(0.01)
    data_4=data_3[data_3['Year']>q]
    #sns.distplot(data_4['Year'])
    #plt.show()
    data_cleaned=data_4.reset_index(drop=True)
    print(data_cleaned.describe(include='all'))
    





if __name__ == '__main__':
    main()