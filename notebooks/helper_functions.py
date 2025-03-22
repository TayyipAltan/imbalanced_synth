import pandas as pd  
import numpy as np

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler

class NoiseSampler(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit_resample(self, X, y):
        return self.resample(X, y)
    
    def _get_num_samples(self, y):
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples
    
    def _generate_syn_data(self, X_minority, num_samples):
        
        sample_results = {}

        for col in X_minority.columns:
            col_min = X_minority[col].min()
            col_max = X_minority[col].max()
            
            s = np.random.uniform(col_min, col_max, size = num_samples)
            
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
        X_sampled = pd.concat([X_train, X_upsampled], axis = 0)

        # Also updating the target variable
        y_sampled = pd.concat([y_train, pd.Series(np.ones(num_samples))], axis = 0)

        return X_sampled, y_sampled


class ColumnScaler(BaseEstimator, TransformerMixin):
    """Transform columns with different scales for SMOTE"""
    
    def __init__(self, cols_to_normalize = None):
        self.cols_to_normalize = cols_to_normalize if cols_to_normalize else ['Time', 'Amount']
        self.scalers = {}
    
    def fit(self, X, y = None):
        self.scalers = {col: MinMaxScaler().fit(X[[col]]) for col in self.cols_to_normalize}
    
    def transform(self, X, y = None):
        X_train = X.copy()
        
        for col in self.cols_to_normalize:
            X_train[col] = self.scalers[col].transform(X_train[col]) 
            
        return X_train  
    
    
# Recode this to inherit from a BaseSampler class
# Then, also adjust the NoiseSampler
class SDVSampler(BaseEstimator):  
    
    def __init__(self, generator = None):
        self.generator = generator
    
    def fit_resample(self, X, y):
        return self.resample(X, y)
    
    def _get_num_samples(self, y): # move this to init
        y_values = y.value_counts()
        num_samples = y_values[0] - y_values[1]
        
        return num_samples
    
    def _generate_syn_data(self, X_minority, num_samples):
        
        X_resampled = X_minority.copy()
        
        # Define metadata and synthesizer with default parameters and no constraints
        metadata = Metadata.detect_from_dataframe(X_resampled)
        synthesizer = self.generator(metadata)
        
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
        X_sampled = pd.concat([X_train, X_upsampled], axis = 0)

        # Also updating the target variable
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

