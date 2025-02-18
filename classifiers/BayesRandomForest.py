from classifiers.NaiveBayes import NaiveBayes
import pandas as pd

class BayesRandomForest:
    def __init__(self, n_classifiers, n_features, n_samples):
        self.classifiers = []
        self.n_classifiers = n_classifiers
        self.n_features = n_features
        self.n_samples = n_samples

    def fit(self, x_train, y_train):
        for _ in range(self.n_classifiers):
            df = pd.concat([x_train, y_train], axis=1)
            sample = df.sample(n=self.n_samples, replace=True)
            x_sample = sample.drop(columns=[y_train.name])
            y_sample = sample[y_train.name]
            self.classifiers.append(self.build_classifier(x_sample, y_sample, self.n_features))

    def predict(self, sample):
        predictions = []
        for classifier, selected_columns in self.classifiers:
            predict_columns = sample[selected_columns]
            predictions.append(classifier.predict(predict_columns))
        
        return max(set(predictions), key=predictions.count)
    
    def predict_proba(self, sample):
        predictions_probability = []
        for classifier, selected_columns in self.classifiers:
            predict_columns = sample[selected_columns]
            predictions_probability.append(classifier.predict_proba(predict_columns))
        
        sums = {}
        for _, predict_proba in predictions_probability:
            for key in predict_proba.keys():
                sums[key] = sums.get(key, 0) + predict_proba[key]

        n_cls = len(predictions_probability)

        predictions = [prediction for prediction, _ in predictions_probability]
        return max(set(predictions), key=predictions.count), {key: value/n_cls for key, value in sums.items()}

    @staticmethod    
    def build_classifier(x_train, y_train, n_features):

        train_columns = x_train.sample(n=n_features, axis=1, replace=True)

        selected_columns = train_columns.columns.tolist()

        bayes = NaiveBayes()
        bayes.fit(train_columns, y_train)

        return (bayes, selected_columns)
