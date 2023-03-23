import pandas as pd
import joblib

class Utils:
    # Loading class
    def load_from_csv(self, path):
        return pd.read_csv(path)
    
    def load_from_mysql(self):
        pass
    # EDA process class
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X,y
    
    # Export model class
    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, 'models/best_model.pkl')