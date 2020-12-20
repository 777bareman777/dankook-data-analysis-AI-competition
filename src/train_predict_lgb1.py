#!/usr/bin/env python
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

import lightgbm as lgbm


def train_predict(train_file, test_file, feature_map_file, predict_valid_file, predict_test_file,
                feature_imp_file, num_threads, metric, learning_rate, boosting,
                objective, verbosity, num_boost_round, early_stopping_rounds, verbose_eval,
                device_type=None, gpu_use_dp=None):

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


    if device_type==None or gpu_use_dp==None:
        device_type='cpu'
        gpu_use_dp=False

    lgbm_params = {
            'num_threads': num_threads,
            'metric': metric,
            'learning_rate': learning_rate,
            'boosting': boosting,
            'objective': objective,
            'num_class': N_CLASS,
            'random_state': SEED,
            'device_type': device_type,
            'gpu_use_dp': gpu_use_dp,
            'verbosity': verbosity,
            }
    
    oof_pred = np.zeros((X.shape[0], N_CLASS))
    test_pred = np.zeros((X_tst.shape[0], N_CLASS))
    n_bests= []

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X,y),1):
        logging.info(f'Training model #{fold}')

        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        dtrn = lgbm.Dataset(X_trn, label=y_trn)
        dval = lgbm.Dataset(X_val, label=y_val)

        logging.info('Training with early stopping')
        lgbm_clf = lgbm.train(params=lgbm_params, train_set=dtrn, num_boost_round=num_boost_round, valid_sets=[dtrn, dval], early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)

        n_best = lgbm_clf.best_iteration
        n_bests.append(n_best)
        logging.info('best iteration={}'.format(n_best))

        test_pred += lgbm_clf.predict(X_tst) / (N_FOLD)
        oof_pred[val_idx] += lgbm_clf.predict(X_val)
        logging.info(f'CV #{fold}: {accuracy_score(y_val, np.argmax(oof_pred[val_idx], axis=1)) * 100:.4f}%')

    imp = pd.DataFrame({'feature': feature_name,
                        'importance': lgbm_clf.feature_importance(importance_type='gain', iteration=n_best)})
    imp = imp.sort_values('importance').set_index('feature')
    imp.to_csv(feature_imp_file)

    logging.info(f'CV: {accuracy_score(y, np.argmax(oof_pred, axis=1)) * 100:.4f}%')
    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, oof_pred, fmt='%.18f', delimiter=',')

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, test_pred, fmt='%.18f', delimiter=',')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')
    parser.add_argument('--predict-valid-file', required=True, dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True, dest='predict_test_file')
    parser.add_argument('--feature-imp-file', required=True, dest='feature_imp_file')
    parser.add_argument('--num-threads', type=int, required=True, dest='num_threads')
    parser.add_argument('--metric', required=True, dest='metric')
    parser.add_argument('--learning-rate', type=float, required=True, dest='learning_rate')
    parser.add_argument('--boosting', required=True, dest='boosting')
    parser.add_argument('--objective', required=True, dest='objective')
    parser.add_argument('--verbosity', type=int, required=True, dest='verbosity')
    parser.add_argument('--num-boost-round', type=int, required=True, dest='num_boost_round')
    parser.add_argument('--early-stopping-rounds', type=int, required=True, dest='early_stopping_rounds')
    parser.add_argument('--verbose-eval', type=int, required=True, dest='verbose_eval')
    parser.add_argument('--device_type', required=False, dest='device_type')
    parser.add_argument('--gpu-use-dp', type=bool ,required=False, dest='gpu_use_dp')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  feature_map_file=args.feature_map_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  feature_imp_file=args.feature_imp_file,
                  num_threads=args.num_threads,
                  metric=args.metric,
                  learning_rate=args.learning_rate,
                  boosting=args.boosting,
                  objective=args.objective,
                  verbosity=args.verbosity,
                  num_boost_round=args.num_boost_round,
                  early_stopping_rounds=args.early_stopping_rounds,
                  verbose_eval=args.verbose_eval,
                  device_type=args.device_type,
                  gpu_use_dp=args.gpu_use_dp)
    logging.info(f'finished ({(time.time() - start) / 60:.2f} min elasped)')

