from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn
import joblib
import os
import argparse
import pandas as pd
import numpy as np

def get_model(model_path):
    return joblib.load(os.path.join(model_path, 'model.joblib'))

if __name__ == '__main__':
    
    print('[INFO] Extracting Arguments')
    print()
    parser = argparse.ArgumentParser()

    # Add hyperparameters that can be sent by user
    parser.add_argument('--random_state', type=int, default=0)

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default='train.csv')
    parser.add_argument('--test-file', type=str, default='test.csv')

    args, _ = parser.parse_known_args()

    print('USING:')
    print(f'SKLearn version: {sklearn.__version__}')
    print(f'Joblib version: {joblib.__version__}')

    print('[INFO] Reading Data')
    print()

    train = pd.read_csv(os.path.join(args.train, args.train_file))
    test = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train.columns)
    target = features.pop(-1)

    print('Building train and test datasets')

    x_train = train[features]
    x_test = test[features]
    y_train = train[target]
    y_test = test[target]

    print('Training Decision Tree Model')

    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(x_train, y_train)

    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)

    print(f'Model is created at {model_path}')

    y_pred = model.predict(x_test)

    print('MODEL METRICS RESULT')
    print(classification_report(y_test, y_pred))
