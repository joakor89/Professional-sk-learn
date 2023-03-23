
# Import Packages
import pandas as pd
import sklearn 
import matplotlib.pyplot as plt 

# Machine learning Modules

#Dimension reduction algorithm
from sklearn.decomposition import PCA 
from sklearn.decomposition import IncrementalPCA 

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
    
    # Table assistance or consultation
    print(X_train.shape)
    print(y_train.shape) 
    
    # We invoke and configure the algorithm pca-consultation
    '''
    The component's number is optional, by default 
    if any number of components is not provided, then, it will 
    assign it this way -> a: n_components = min(n_samples, n_features)
    '''
    # The component's number expected, Let's set it up:
    pca = PCA(n_components=3)
    
    # PCA the train model adjustment procedure
    pca.fit(X_train)
    
    # Since a IncrementalPCA contrast is required, then, let's
    # applied a similar procedure ran out above
    # the additional parameter correspond to "IPCA" training call
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)
    
    '''
    Let's plot from 0 to the PCA suggested component's length  
    x-axis Vs. y-axis & the value of importance in each of these
    components. Thus, it can identify which are really important for
    the model
    '''
    
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()
    
    # Now! let's set-up the Logistic Regression environment
    # let's allocate the solver in order to avoid future warnings & mistakes
    logistic = LogisticRegression(solver='lbfgs')
    
    # Training data configure
    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    # Now! lets invoke the Logisitic Regression to adjust both datasets
    logistic.fit(dt_train, y_train)
    # Let's measure the prediction's accuracy
    print("PCA Score: ", logistic.score(dt_test,y_test))
    
    # Let's perform a similar procedure but invoking IPCA
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    # Now! lets invoke the Logisitic Regression to adjust both datasets with IPCA
    logistic.fit(dt_train, y_train) 
    # Let's measure the prediction's accuracy 
    print("IPCA Score: ", logistic.score(dt_test,y_test)) 
    
    
