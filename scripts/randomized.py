# Import Packages
import pandas as pd
import warnings
warnings.simplefilter("ignore")


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dataset = pd.read_csv('data/happiness.csv')
    
    # Let's check the dataset
    print(dataset)
    
    X = dataset.drop(['country', 'rank', 'score'], axis=1)
    y = dataset[['score']]
    
    # Let's define the regresson model to use
    reg = RandomForestRegressor()
    
    # Let's define the optimazier Grid to use
    parameters = {
        # lets the regressor estimated within this range by default
        'n_estimators' : range(4, 16),
        # Let's see the split made by the tree as a quality measurement
        'criterion' : ['friedman_mse', 'absolute_error'],
        # Tree's Depth
        'max_depth' : range(2, 11)
    }
    
    # Let's define the random estimator & adjust directly the model
    rand_est = RandomizedSearchCV(reg, parameters, n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X, y)
    
    print("="*64)
    #Let's check the best estimator
    print(rand_est.best_estimator_)
    # Let's also check the chosen parameters
    print(rand_est.best_params_)
    
    '''
    Since it's a pseudo-random process! 
    it is normal to obtain different outcomes
    
    '''
    
    # Let's perform a simple prediction
    print("="*64)
    print('The Happiness index for the first country: ', rand_est.predict(X.loc[[0]]))
    
    
    