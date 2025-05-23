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
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def _get_num_samples(self, y):
        """Get the number of samples to generate for the minority class."""
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples
    
    
    def resample(self, X, y):
        """Resample the data using the specified sampling method. Should be the same as the original ones"""
        raise NotImplementedError("Subclasses should implement this method.")

class NoiseSampler(BaseEstimator, BaseSampler): # Do I still need BaseEstimator if I inherit from BaseSampler?
    
    def __init__(self, random_state = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit_resample(self, X, y):
        return self.resample(X, y)
    
    def _get_num_samples(self, y):
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples
    
    def _generate_syn_data(self, X_minority, num_samples):
        
        sample_results = {}

        for col in X_minority.columns:
            col_mean = X_minority[col].mean()
            col_std = X_minority[col].std()
            
            s = np.random.normal(loc=col_mean, scale = col_std, size = num_samples)
            
            sample_results[col] = s

        X_upsampled = pd.DataFrame(sample_results)
        
        return X_upsampled
    
    def resample(self, X, y):
        """Function to generate synthetic samples for the minority class using noise"""
        
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
    
    
# Recode this to inherit from a BaseSampler class
# Then, also adjust the NoiseSampler
class SDVSampler(BaseEstimator):  
    
    def __init__(self, generator, metadata):
        
        self.generator = generator
        self.metadata = metadata
    
    def fit_resample(self, X, y):
        return self.resample(X, y)
    
    def _get_num_samples(self, y): # move this to init
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples
    
    def _generate_syn_data(self, X_minority, num_samples):
        
        X_resampled = X_minority.copy()
        
        # Creating a new instance of the synthesizer within each fold
        synthesizer = self.generator(self.metadata)
        
        # Fitting and resampling
        synthesizer.fit(X_resampled)
        X_sds = synthesizer.sample(num_samples)
        
        return X_sds
    
    def resample(self, X, y):
        """Function to generate synthetic samples for the minority class using noise"""
        
        X_train = X.copy()
        y_train = y.copy()
        
        num_samples = self._get_num_samples(y_train)

        # Obtain the minority class that needs to be upsampled
        X_minority = X_train[y_train == 1]
        
        # Generate synthetic samples
        X_upsampled = self._generate_syn_data(X_minority, num_samples)

        # Also updating the target variable
        X_sampled = pd.concat([X_train, X_upsampled], axis = 0)
        y_sampled = pd.concat([y_train, pd.Series(np.ones(num_samples))], axis = 0)

        return X_sampled, y_sampled         
            
            
def get_scores(y_test, y_preds, sampling_procedures):
    """Calculates precison, recall, F1, and ROC AUC scores for each sampling procedure.
    """
    scores_dict = {}
    
    for i in range(len(y_preds)):
        precision = precision_score(y_test, y_preds[i], average = 'macro')
        recall = recall_score(y_test, y_preds[i], average = 'macro')
        f1 = f1_score(y_test, y_preds[i], average = 'macro')
        roc_auc = roc_auc_score(y_test, y_preds[i])
        
        scores_dict[sampling_procedures[i]] = [precision, recall, f1, roc_auc]
    
    scores_df = pd.DataFrame(scores_dict).T
    scores_df.columns = ['Precision', 'Recall', 'F1', 'ROC_AUC']
    
    return scores_df

