# Import Packages
import pandas as pd
import warnings
warnings.simplefilter("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

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
    
    '''
    I've decided to implement another classifier which
    is as not efficient by itself as many other, in order
    to contrast its performance with other ones
    '''
    
    # Let's set up and invoke the classifier directly adjusted
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    # Let's now perform our predictions
    knn_pred = knn_class.predict(X_test)
    # Let's print the outcome
    print("="*64)
    print(accuracy_score(knn_pred, y_test))
    
    # Let's set up and invoke the classifier directly adjusted
    bag_class = BaggingClassifier(base_estimator = KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    # Let's now perform our predictions
    bag_pred = bag_class.predict(X_test)
    # Let's print the outcome
    print("="*64)
    print(accuracy_score(bag_pred, y_test)) 
    
    
    