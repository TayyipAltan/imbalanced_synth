import pandas as pd  
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import EditedNearestNeighbours

class BaseSampler(BaseEstimator):
    """Base class for all samplers. This class is not meant to be used directly.
    """
    
    def __init__(self, random_state = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X, y):
        """Fit the sampler to the data and return the resampled data"""
        return self.resample(X, y)
    
    def _get_num_samples(self, y):
        """Get the number of samples to generate for the minority class."""
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples  

    def resample(self, X, y):
        """Resample the data using the specified sampling method"""
        
        X_train = X.copy()
        y_train = y.copy()
        
        num_samples = self._get_num_samples(y_train)

        # Obtain the minority class that needs to be upsampled
        X_minority = X_train[y_train == 1]
        
        # Generate synthetic samples
        X_upsampled = self._generate_syn_data(X_minority, num_samples)

        # pdating indepent variables and the target variable
        X_sampled = pd.concat([X_train, X_upsampled], axis = 0)
        y_sampled = pd.concat([y_train, pd.Series(np.ones(num_samples))], axis = 0)

        return X_sampled, y_sampled
   
    
class NoiseSampler(BaseSampler):
    
    def __init__(self, random_state = None):
        super().__init__(random_state)
           
    def _generate_syn_data(self, X_minority, num_samples):
        
        sample_results = {}

        for col in X_minority.columns:
            col_mean = X_minority[col].mean()
            col_std = X_minority[col].std()
            
            s = np.random.normal(loc=col_mean, scale = col_std, size = num_samples)
            
            sample_results[col] = s

        X_upsampled = pd.DataFrame(sample_results)
        
        return X_upsampled    


class SDVSampler(BaseSampler):  
    
    def __init__(self, generator, metadata, random_state = None):
        super().__init__(random_state)
        self.generator = generator
        self.metadata = metadata
       
    def _generate_syn_data(self, X_minority, num_samples):
        
        X_resample = X_minority.copy()
        
        # Creating a new instance of the synthesizer within each fold
        synthesizer = self.generator(self.metadata)

        synthesizer.fit(X_resample)
        X_sds = synthesizer.sample(num_samples)
        
        return X_sds
   
    
class SDVENN(SDVSampler):  
    
    def __init__(self, generator, metadata, random_state=None):
        super().__init__(generator, metadata, random_state)
        self.enn = EditedNearestNeighbours()

    def resample(self, X, y):
        
        # Oversample using the original SDV class
        X_sampled, y_sampled = super().resample(X, y)
        
        X_resampled, y_resampled = self.enn.fit_resample(X_sampled, y_sampled)

        return X_resampled, y_resampled
        
               
def display_scores(cv_scores, scorings):
    
    res = {}
   
    for scoring in scorings:
        res[scoring.capitalize()] = [round(score[f"test_{scoring}"].mean(), 3) for score in cv_scores]
    
    res_df = pd.DataFrame(res)
    res_df.index = ['Noise', 'ROS', 'SMOTE', 'CTGAN', 'TVAE', 'SMOTENN', 
                    'CTGANENN', 'CSL']
    
    return res_df
       

