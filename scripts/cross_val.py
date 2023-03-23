# Import Packages
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dataset = pd.read_csv('data/happiness.csv')
    
    # Let's know safe the dataset 
    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset['score']
    
    # Let's define the model and store it
    model =DecisionTreeRegressor()
    # Let's perform a quick diagnose with cross validation score
    score = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
    print("="*64)
    # This can be printed directly or apply numpy on it to condensate all previous outcomes
    print(np.mean(score))
    # Regularly, it's mainly used for scoring is the absolute value
    print("="*64)
    print("With Abs Value: ", np.abs(np.mean(score)))
    
    '''
    To see how cross-validation perform behind scenes, then, Kfold() function
    
    '''
    
    # K-fold inner performance
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train, test in kf.split(dataset):
        print(train)
        print(test)
    
    