# By agreed conventions, Set-up a entry point to summarize inner performace
# Import packages
from utils import Utils
from models import Models

if __name__ == '__main__':
    
    utils = Utils()
    models = Models()
    
    data = utils.load_from_csv('in/happiness.csv')
    X, y = utils.features_target(data, ['score', 'rank', 'country'], ['score'])
    
    models.grid_training(X,y)
    
    print(data)