# Import Packages
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    # Loading Data Frame with Pandas
    dataset = pd.read_csv('data/candy.csv')
    
    # Having the first ten register
    print(dataset.head(10))
    
    '''
    Since this Diagnose exploration is a unsupervised learning model, there is no need to
    broke the dataset down, instead, I've decided to send it directly to K-means process
    '''
    
    # Let's know safe the dataset 
    X = dataset.drop(['competitorname'], axis=1)
    # Now! Let's invoke the batch & pass its arguments by
    # number desired of cluster 4
    # batch will setting groups by 8 datapoint each & adjust it
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    # Let's check how many center the algorithm generates
    print("Total of centers:", len(kmeans.cluster_centers_))
    print("="*64)
    # Let's now make a prediction & check how it'll label the data
    print(kmeans.predict(X))
    
    #Let's integrate K-Means algorithm with the rest of the original dataset
    # Let's add a new columns containing the cluster group predicted on kmeans
    dataset['group'] = kmeans.predict(X)
    # Let's check on it
    print(dataset)
    
    