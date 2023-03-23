# Import Packages
import pandas as pd
import sklearn 

# Machine learning Modules
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Now! let's import the training metrics & the Mean Squared Error MSE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Entry point
if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dataset = pd.read_csv('data/main-2017.csv')
    
    # Let's get the statistical report
    print(dataset.describe())
    
    # Now! Let's manually select the features
    X = dataset[['GDP', 'Family', 'Lifexp', 'Freedom', 'corruption', 'Generosity', 'Dystopia']]
    # Let's define target, but only with the score column
    y = dataset[['Score']]
    
    # Table assistance or consultation
    print(X.shape)
    print(y.shape) 
    
    # Let's split the training set. 
    #  & add a test size of 25 %
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    # Now! let's set-up the Linear Regression environment
    # let's allocate the solver in order to avoid future warnings & mistakes
    modelLinear = LinearRegression().fit(X_train, y_train)
    # Let's calculate the prediction
    y_predict_linear = modelLinear.predict(X_test)
    
    # Now! let's set-up the Lasso Regression environment
    # alpha is the lambda contain on its math formula
    modelLasso = Lasso(alpha = 0.2).fit(X_train, y_train)
    # Let's calculate the prediction
    y_predict_lasso = modelLasso.predict(X_test) 
    
    # Now! let's set-up the Ridge Regression environment
    # alpha is the lambda contain on its math formula
    modelRidge = Lasso(alpha = 1).fit(X_train, y_train)
    # Let's calculate the prediction
    y_predict_ridge = modelRidge.predict(X_test)  
    
    '''
    We have calculated the loss for each of the models trained,
    Now! let's start with the linear model, with the mean 
    square error and apply it to the test data with the prediction made.
    '''
    
    # Let's set up the loss on each model performed so far
    
    # Linear Model
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print('Linear_loss: ', linear_loss)
    # Lasso Model
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print('Lasso_loss: ', lasso_loss)
    # Ridge Model
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print('Ridge_loss: ', ridge_loss)
    
    # Let's set a boundary
    print("="* 32)
    # Let's print the coefficients to see how it affects each of the regressions
    print("Lassos's Coefficient")
    print(modelLasso.coef_)
    
    # Let's perform the same procedure with Ridge
    print("="* 32)
    print("Ridge's Coefficient")
    print(modelRidge.coef_) 
    
    
    
