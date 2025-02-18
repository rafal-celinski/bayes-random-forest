import pandas as pd
from sklearn.tree import DecisionTreeClassifier
  
class ClassicRandomForest:
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
            predictions.append(classifier.predict(predict_columns)[0])
        
        return max(set(predictions), key=predictions.count)
    
    def predict_proba(self, sample):
        probabilities = []
        predictions = []
        for classifier, selected_columns in self.classifiers:
            predict_columns = sample[selected_columns]
            predictions.append(classifier.predict([predict_columns])[0])
            probability = classifier.predict_proba([predict_columns])[0]
            probabilities.append({class_label: prob for class_label, prob in enumerate(probability)})
        
        sums = {}
        for predict_proba in probabilities:
            for key in predict_proba.keys():
                sums[key] = sums.get(key, 0) + predict_proba[key]

        n_cls = len(probabilities)

        return max(set(predictions), key=predictions.count), {key: value/n_cls for key, value in sums.items()}

    @staticmethod    
    def build_classifier(x_train, y_train, n_features):

        train_columns = x_train.sample(n=n_features, axis=1, replace=True)

        selected_columns = train_columns.columns.tolist()

        tree = DecisionTreeClassifier()
        tree.fit(train_columns.values, y_train)

        return (tree, selected_columns)

