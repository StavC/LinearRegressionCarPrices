import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler




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

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))  # sharey -> share 'Price' as y
    ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
    ax1.set_title('Price and Year')
    ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
    ax2.set_title('Price and EngineV')
    ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
    ax3.set_title('Price and Mileage')

    plt.show()
        # From the subplots and the PDF of price, we can easily determine that 'Price' is exponentially distributed
        # A good transformation in that case is a log transformation
    #sns.distplot(data_cleaned['Price'])
         # Let's transform 'Price' with a log transformation
    log_price = np.log(data_cleaned['Price'])

         # Then we add it to our data frame
    data_cleaned['log_price'] = log_price

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
    ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
    ax1.set_title('Log Price and Year')
    ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
    ax2.set_title('Log Price and EngineV')
    ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
    ax3.set_title('Log Price and Mileage')
         # The relationships show a clear linear relationship
         # This is some good linear regression material
         # Alternatively we could have transformed each of the independent variables
    plt.show()

    ###### check for multicollinearity
    # To make this as easy as possible to use, we declare a variable where we put
    # all features where we want to check for multicollinearity
    variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
    vif = pd.DataFrame()
    # here we make use of the variance_inflation_factor, which will basically output the respective VIFs
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    # Finally, I like to include names so it is easier to explore the result
    vif["Features"] = variables.columns
    print(vif)
    # Since Year has the highest VIF, remove it from the model
    # This will drive the VIF of other variables down!!!
    # So even if EngineV seems with a high VIF, too, once 'Year' is gone that will no longer be the case
    data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
    variables = data_no_multicollinearity[['Mileage', 'EngineV']]
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif["Features"] = variables.columns
    print(vif)
    # the VIF is low for Mileage and EngineV

        #### create Dummy Variables
    data_with_dummys=pd.get_dummies(data_no_multicollinearity,drop_first=True)
    print(data_with_dummys.head())
    # To make the code a bit more parametrized, let's declare a new variable that will contain the preferred order

    cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
            'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
            'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
            'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
            'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']
    data_preprocessed = data_with_dummys[cols]

    ### geting VIF for all features
    # Let's simply drop log_price from data_preprocessed
    variables = data_preprocessed.drop(['log_price'], axis=1)
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif["features"] = variables.columns
    print(vif)


    ###### Linear Regerssion Model
    # The target(s) (dependent variable) is 'log price'
    targets = data_preprocessed['log_price']

    # The inputs are everything BUT the dependent variable, so we can simply drop it
    inputs = data_preprocessed.drop(['log_price'], axis=1)


if __name__ == '__main__':
    main()