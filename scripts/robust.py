# Import Packages
import pandas as pd
import warnings
warnings.simplefilter("ignore")

from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dataset = pd.read_csv('data/corrupt.csv')
    
    # Having the first five register
    print(dataset.head(5))
    
    # Let's know safe the X dataset 
    X = dataset.drop(['country', 'score'], axis=1)
    # The dataset just with the target column
    y = dataset['score']
    
     # Let's split the training set. 
    # To add replicability we use the random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    estimators = {
        # SVR Estimator
        'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1),
        # RANSAC META-ESTIMATOR
        'RANSAC': RANSACRegressor(),
        'Huber' : HuberRegressor(epsilon=1.35)
    }
    
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        
        print("="*64)
        print(name)
        print('MSE: ', mean_squared_error(y_test, predictions))
        
