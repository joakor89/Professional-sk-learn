# Import Packages
import pandas as pd
import warnings
warnings.simplefilter("ignore")

# Let's now set the model up on Decision-Trees algorithm
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dt_heart = pd.read_csv('data/heart.csv')
    # A quick statistical review
    print(dt_heart['target'].describe())
    
    # Let's know safe the X dataset 
    X = dt_heart.drop(['target'], axis=1)
    # The dataset just with the target column
    y = dt_heart['target']
    
    # Let's split the training set. 
    # To add replicability we use the random state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
    
    # Let's set up and invoke the classifier directly adjusted
    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    # Let's now perform our predictions
    boost_pred = boost.predict(X_test)
    
    # Let's print the outcome
    print("="*64)
    print(accuracy_score(boost_pred, y_test))
    
    
    
    