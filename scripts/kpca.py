# Import Packages
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt 

# Machine learning Modules

#Dimension reduction algorithm
from sklearn.decomposition import KernelPCA

# Linear Classifier
from sklearn.linear_model import LogisticRegression 

#Pre-processing Modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dt_heart = pd.read_csv('data/heart.csv')
    
    # Having the first five register
    print(dt_heart.head(5))
    
    # Let's know safe the dataset without the target column
    dt_features = dt_heart.drop(['target'], axis=1)
    # The dataset with just the column previously dropped
    dt_target = dt_heart['target']
    
    # Now! Let's normalize the data with Standard Scaler
    dt_features = StandardScaler().fit_transform(dt_features)
    
    # Let's split the training set. 
    # To add replicability we use the random state
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
    
    #Let's define a KPCA variable 
    kpca = KernelPCA(n_components=4, kernel='poly')
    
    #Let's adjust the data
    kpca.fit(X_train)
    
     # Training data configure
    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test) 
    
        # Now! let's set-up the Logistic Regression environment
    # let's allocate the solver in order to avoid future warnings & mistakes
    logistic = LogisticRegression(solver='lbfgs')
    
    # Now! lets invoke the Logisitic Regression to adjust both datasets
    logistic.fit(dt_train, y_train)
    print("KPCA Score: ", logistic.score(dt_test,y_test))
    
    
    
    
    