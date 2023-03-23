# Import Packages
import pandas as pd
# Invoke MeanShift from sklearn
from sklearn.cluster import MeanShift

# Let's set up the Entry Point
if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dataset = pd.read_csv('data/candy.csv')
    
    # Having the first five register
    print(dataset.head())
   
    # Let's know safe the dataset 
    X = dataset.drop('competitorname', axis=1)
    
     # Let's define a variable to store the model
    '''
    I've decided not allocated any parameter and 
    let the algorithm processthe matemathical perfomance by itself
    & finally adjust the model
    '''
    meanshift = MeanShift().fit(X)
    # Let's check the labels to classify used by the algorithm
    print(max(meanshift.labels_))
    print("=="*64)
    # Let's check the center data location
    print(meanshift.cluster_centers_)
    
    # Let's now integrated within the dataset
    # Let's created a new column to store the info
    dataset['meanshift'] = meanshift.labels_
    print(dataset)
    
    '''
    The outcome ought to be contrast it with others algortihm's performance such as K-Means
    & evaluate which one perform the better: "computational timing", labels allocation and
    so on...
    '''
    
    
    
   
    
   
