from sklearn.base import BaseEstimator, ClassifierMixin

class KDEClassifier(BaseEstimator, ClassifierMixin):
	"""Bayesian generative classification based on KDE
	Parameters
	----------
	bandwidth : float
	the kernel bandwidth within each class
	kernel : str
	the kernel name, passed to KernelDensity
	"""
	def __init__(self, bandwidth=1.0, kernel='gaussian'):
		self.bandwidth = bandwidth
		self.kernel = kernel
        
	def fit(self, X, y):
		self.classes_ = np.sort(np.unique(y))
		training_sets = [X[y == yi] for yi in self.classes_]
		self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
		self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
		return self
    
	def predict_proba(self, X):
		logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
		result = np.exp(logprobs + self.logpriors_)   
		return result / result.sum(1, keepdims=True)
    
	def predict(self, X):
		return self.classes_[np.argmax(self.predict_proba(X), 1)]	