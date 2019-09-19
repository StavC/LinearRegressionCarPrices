import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split





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

   # f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))  # sharey -> share 'Price' as y
    #ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
   # ax1.set_title('Price and Year')
    #ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
    #ax2.set_title('Price and EngineV')
    #ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
    #ax3.set_title('Price and Mileage')

    #plt.show()
        # From the subplots and the PDF of price, we can easily determine that 'Price' is exponentially distributed
        # A good transformation in that case is a log transformation
    #sns.distplot(data_cleaned['Price'])
         # Let's transform 'Price' with a log transformation
    log_price = np.log(data_cleaned['Price'])

         # Then we add it to our data frame
    data_cleaned['log_price'] = log_price

   # f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 3))
   # ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
    #ax1.set_title('Log Price and Year')
    #ax2.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
    #ax2.set_title('Log Price and EngineV')
    #ax3.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
    #ax3.set_title('Log Price and Mileage')
         # The relationships show a clear linear relationship
         # This is some good linear regression material
         # Alternatively we could have transformed each of the independent variables
    #plt.show()

    ###### check for multicollinearity
    # To make this as easy as possible to use, we declare a variable where we put
    # all features where we want to check for multicollinearity
    variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
    vif = pd.DataFrame()
    # here we make use of the variance_inflation_factor, which will basically output the respective VIFs
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    # Finally, I like to include names so it is easier to explore the result
    vif["Features"] = variables.columns
    #print(vif)
    # Since Year has the highest VIF, remove it from the model
    # This will drive the VIF of other variables down!!!
    # So even if EngineV seems with a high VIF, too, once 'Year' is gone that will no longer be the case
    data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
    variables = data_no_multicollinearity[['Mileage', 'EngineV']]
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
    vif["Features"] = variables.columns
    #print(vif)
    # the VIF is low for Mileage and EngineV

        #### create Dummy Variables
    data_with_dummys=pd.get_dummies(data_no_multicollinearity,drop_first=True)
    #print(data_with_dummys.head())
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
    #print(vif)


    ###### Linear Regerssion Model
    # The target(s) (dependent variable) is 'log price'
    targets = data_preprocessed['log_price']

    # The inputs are everything BUT the dependent variable, so we can simply drop it
    inputs = data_preprocessed.drop(['log_price'], axis=1)
    # Create a scaler object
    scaler = StandardScaler()
    # Fit the inputs (calculate the mean and standard deviation feature-wise)
    scaler.fit(inputs)
    # Scale the features and store them in a new variable (the actual scaling procedure)
    inputs_scaled = scaler.transform(inputs)

    ### TRAIN TEST SPLIT
    x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train, y_train) #X the values,Y the right log price
    # Let's check the outputs of the regression
    # I'll store them in y_hat as this is the 'theoretical' name of the predictions
    y_hat = reg.predict(x_train)
    # The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
    # The closer the points to the 45-degree line, the better the prediction
    plt.scatter(y_train, y_hat)
    # Let's also name the axes
    plt.xlabel('Targets (y_train)', size=18)
    plt.ylabel('Predictions (y_hat)', size=18)
    # Sometimes the plot will have different scales of the x-axis and the y-axis
    # This is an issue as we won't be able to interpret the '45-degree line'
    # We want the x-axis and the y-axis to be the same
    plt.xlim(6, 13)
    plt.ylim(6, 13)
    plt.show()
    # Another useful check of our model is a residual plot
    # We can plot the PDF of the residuals and check for anomalies
    sns.distplot(y_train - y_hat)

    # Include a title
    plt.title("Residuals PDF", size=18)

    # In the best case scenario this plot should be normally distributed
    # In our case we notice that there are many negative residuals (far away from the mean)
    # Given the definition of the residuals (y_train - y_hat), negative values imply
    # that y_hat (predictions) are much higher than y_train (the targets)
    # This is food for thought to improve our model
    # Find the R-squared of the model
    reg.score(x_train, y_train)

    # Note that this is NOT the adjusted R-squared
    # in other words... find the Adjusted R-squared to have the appropriate measure :)
    # Obtain the bias (intercept) of the regression
    print(reg.intercept_)
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    print(reg_summary)
    # Check the different categories in the 'Brand' variable
    data_cleaned['Brand'].unique()
    # In this way we can see which 'Brand' is actually the benchmark-AUDI
    ### TESTING
    # Once we have trained and fine-tuned our model, we can proceed to testing it
    # Testing is done on a dataset that the algorithm has never seen
    # Luckily we have prepared such a dataset
    # Our test inputs are 'x_test', while the outputs: 'y_test'
    # We SHOULD NOT TRAIN THE MODEL ON THEM, we just feed them and find the predictions
    # If the predictions are far off, we will know that our model overfitted
    y_hat_test = reg.predict(x_test)

    # Create a scatter plot with the test targets and the test predictions
    # You can include the argument 'alpha' which will introduce opacity to the graph
    plt.scatter(y_test, y_hat_test, alpha=0.2)
    plt.xlabel('Targets (y_test)', size=18)
    plt.ylabel('Predictions (y_hat_test)', size=18)
    plt.xlim(6, 13)
    plt.ylim(6, 13)
    plt.show()

    #Finally, let's manually check these predictions
    # To obtain the actual prices, we take the exponential of the log_price
    df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
    # Therefore, to get a proper result, we must reset the index and drop the old indexing
    y_test = y_test.reset_index(drop=True)
    df_pf['Target'] = np.exp(y_test)

    # Additionally, we can calculate the difference between the targets and the predictions
    # Note that this is actually the residual (we already plotted the residuals)
    df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

    # Since OLS is basically an algorithm which minimizes the total sum of squared errors (residuals),
    # this comparison makes a lot of sense
    # Finally, it makes sense to see how far off we are from the result percentage-wise
    # Here, we take the absolute difference in %, so we can easily order the data frame
    df_pf['Difference%'] = np.absolute(df_pf['Residual'] / df_pf['Target'] * 100)
    # Sometimes it is useful to check these outputs manually
    # To see all rows, we use the relevant pandas syntax
    pd.options.display.max_rows = 999
    # Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    # Finally, we sort by difference in % and manually check the model
    df_pf=df_pf.sort_values(by=['Difference%'])
    print(df_pf)

if __name__ == '__main__':
    main()