#!/usr/bin/evn python
import argparse
import logging
import os
import time
from kaggler.data_io import load_data
from const import N_FOLD, N_CLASS, SEED

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

def train_predict(train_file, test_file, feature_map_file, predict_valid_file, predict_test_file,
                    feature_imp_file, n_jobs, n_estimators, max_features, verbose):

    model_name = os.path.splitext(os.path.splitext(os.path.basename(predict_test_file))[0])[0]

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename=f'{model_name}.log')

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)

    with open(feature_map_file) as f:
        feature_name = [x.strip() for x in f.readlines()]

    logging.info('Loading CV Ids')
    cv = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    rf_params = {
            'n_jobs': n_jobs,
            'n_estimators': n_estimators,
            'max_features': max_features,
            'verbose': verbose,
            'random_state': SEED,
            }

    oof_pred = np.zeros((X.shape[0], N_CLASS))
    test_pred = np.zeros((X_tst.shape[0], N_CLASS))

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X,y),1):
        logging.info(f'Training model #{fold}')
        
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        rf_clf = RandomForestClassifier(**rf_params)
        rf_clf.fit(X_trn, y_trn)

        test_pred += rf_clf.predict_proba(X_tst) / N_FOLD
        oof_pred[val_idx] += rf_clf.predict_proba(X_val)
        logging.info(f'CV #{fold}: {accuracy_score(y_val , np.argmax(oof_pred[val_idx], axis=1)) * 100:.4}%')


    imp = pd.DataFrame({'feature': feature_name,
                        'importance': rf_clf.feature_importances_})
    imp = imp.sort_values('importance').set_index('feature')
    imp.to_csv(feature_imp_file)

    logging.info(f'CV: {accuracy_score(y, np.argmax(oof_pred, axis=1)) * 100:.4f}%')
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, oof_pred,fmt='%.18f', delimiter=',')

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, test_pred,fmt='%.18f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--feature-imp-file', required=True, dest='feature_imp_file')
    parser.add_argument('--n-jobs', type=int, required=True, dest='n_jobs')
    parser.add_argument('--n-estimators', type=int, required=True,  dest='n_estimators')
    parser.add_argument('--max-features', required=True, dest='max_features')
    parser.add_argument('--verbose', type=int, required=True, dest='verbose')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_imp_file=args.feature_imp_file,
                  n_jobs=args.n_jobs,
                  n_estimators=args.n_estimators,
                  max_features=args.max_features,
                  verbose=args.verbose)
    logging.info(f'finished ({(time.time() - start) / 60:.2f} min elasped)')
