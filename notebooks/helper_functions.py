import pandas as pd  
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


class BaseSampler(BaseEstimator):
    """Base class for all samplers. This class is not meant to be used directly.
    """
    
    def __init__(self, random_state = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X, y):
        """Fit the sampler to the data and return the resampled data. Essentially the same as what is done in the original code"""
        return self.resample(X, y)
    
    def _get_num_samples(self, y):
        """Get the number of samples to generate for the minority class."""
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples  
    
    def _generate_syn_data(self, X_minority, num_samples):
        raise NotImplementedError("Subclasses should implement `_generate_syn_data`.")

    def resample(self, X, y):
        """Resample the data using the specified sampling method. Should be the same as the original ones"""
        
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


class ColumnScaler(BaseEstimator, TransformerMixin):
    """Transform columns with different scales for SMOTE"""
    
    def __init__(self, cols_to_normalize = None):
        self.cols_to_normalize = cols_to_normalize if cols_to_normalize else ['Time', 'Amount']
        self.scaler = StandardScaler()
    
    def fit(self, X, y = None):
        self.scaler = self.scaler.fit(X[self.cols_to_normalize])
        return self
    
    def transform(self, X, y = None):
        X_train = X.copy()
        
        X_train[self.cols_to_normalize] = self.scaler.transform(X_train[self.cols_to_normalize]) 
            
        return X_train  
    
    
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
    
    def __init__(self, generator, metadata, random_state = None, def_distr = None):
        super().__init__(random_state)
        self.generator = generator
        self.metadata = metadata
        self.def_distr = def_distr 
       
    def _generate_syn_data(self, X_minority, num_samples):
        
        X_resampled = X_minority.copy()
        
        # If statement in case Gaussian Copula is used
        if self.def_distr is not None:
            # Creating a new instance of the synthesizer within each fold
            synthesizer = self.generator(self.metadata, 
                                         default_distribution=self.def_distr)
        else:
            synthesizer = self.generator(self.metadata)
        
        # Fitting and resampling
        synthesizer.fit(X_resampled)
        X_sds = synthesizer.sample(num_samples)
        
        return X_sds
        
               
def display_scores(cv_scores, scorings):
    
    res = {}
   
    for scoring in scorings:
        res[scoring.capitalize()] = [round(score[f"test_{scoring}"].mean(), 3) for score in cv_scores]
    
    res_df = pd.DataFrame(res)
    res_df.index = ['Baseline', 'Noise', 'ROS', 'SMOTE', 'CTGAN', 'TVAE', 'Gaussian Copula',
                    'CSL']
    
    return res_df
       

