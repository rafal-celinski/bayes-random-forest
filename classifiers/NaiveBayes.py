import pandas as pd

class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def fit(self, x_train, y_train):
        self.priors = y_train.value_counts(normalize=True).replace(0, 1e-6)
        df = pd.concat([x_train, y_train], axis=1)
        
        for cls in y_train.unique():
            class_data = df[df[y_train.name] == cls].drop(columns=[y_train.name])
            feature_prob = class_data.apply(lambda col: col.value_counts(normalize=True).replace(0, 1e-6), axis=0)
            self.likelihoods[cls] = feature_prob
        
    def predict(self, sample):
        postori = {}
        for cls, prob in self.priors.items():
            for feature in self.likelihoods[cls]:
                prob *= self.likelihoods[cls][feature].get(sample[feature], 1e-6)
            postori[cls] = prob if prob != 0 else 1e-6
        return max(postori, key=postori.get)
    
    def predict_proba(self, sample):
        postori = {}
        for cls, prob in self.priors.items():
            for feature in self.likelihoods[cls]:
                prob *= self.likelihoods[cls][feature].get(sample[feature], 1e-6)
            postori[cls] = prob if prob != 0 else 1e-6
        
        postori_sum = sum(postori.values())
        return max(postori, key=postori.get), {key: value / postori_sum  for key, value in postori.items()}
