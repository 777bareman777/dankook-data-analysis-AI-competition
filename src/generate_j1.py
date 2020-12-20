#!/usr/bin/env python

import argparse
from kaggler.data_io import save_data
import logging
import numpy as np
import pandas as pd
import time

from const import ID_COL, TARGET_COL

def generate_feature(train_file, test_file,
        train_feature_file, target_label_file, test_feature_file, feature_map_file):
    logging.info('loading raw data')
    
    # 대회 데이터 로드 및 타겟 값 로드 
    trn = pd.read_csv(train_file, index_col=ID_COL)
    tst = pd.read_csv(test_file, index_col=ID_COL)

    # 이상치 제거
    # test의 MinMax 범위 넘는 행은 train에서 제거
    n_trn = trn.shape[0]
    for col in trn.columns[:18]:
        trn = trn.loc[np.logical_and(trn[col] >= tst[col].min(),
                                    trn[col] <= tst[col].max())]

    logging.info(f'Number of rows removed :{n_trn - trn.shape[0]}')
    n_trn = trn.shape[0];y = trn[TARGET_COL].values

    # 데이터셋을 합쳐서 한꺼번에 가공
    trn.drop(TARGET_COL, axis=1, inplace=True)
    dataset = pd.concat([trn,tst], axis=0)
    dataset.fillna(-1, inplace=True)

    # 새로운 feature에 만들때 사용할 필드 선택
    wave_columns = dataset.columns.drop(['nObserve', 'nDetect', 'redshift'])

    # 선택한 필드들을 가지고 앞뒤 간의 차를 이용해서 새로운 변수 생성
    for j in range(14):
        name = 'diff_' + str(wave_columns[j+1]) + '_' + str(wave_columns[j])
        dataset[name] = dataset[wave_columns[j+1]] - dataset[wave_columns[j]]
        logging.info(f'{wave_columns[j+1]} - {wave_columns[j]} {j}')

    # 선택한 필드들을 가지고 15포인트 랭킹 변수 생성
    mag_rank = dataset[wave_columns].rank(axis=1)

    rank_col = []
    for col in trn[wave_columns].columns:
        col = col + '_rank'
        rank_col.append(col)
    mag_rank.columns = rank_col

    dataset = pd.concat([dataset, mag_rank], axis=1)

    # 선택한 필드들을 가지고
    # 측정방법별 파장 차이 비교 변수 생성
    diff_col = []
    for col in ['u','g','r','i','z']:
        for i in range(2):
            diff_col.append(col + '_' + str(i))

    mag_wave = pd.DataFrame(np.zeros((dataset.shape[0],10)), index=dataset.index)

    for i in range(0,10,5):
        for j in range(5):
            mag_wave.loc[:, j+i] = dataset[wave_columns[j]] - dataset[wave_columns[5+j+i]]
            logging.info(f'{wave_columns[j]} - {wave_columns[5+j+i]} {i+j}')

    # 새롭게 만든 변수들을 대회 데이터체 추가
    mag_wave.columns = diff_col
    dataset = pd.concat([dataset, mag_wave], axis=1)

    # 멱함수 분포를 정규 분포를 만들기 위해서, np.log1p를 사용
    # 그리고 nObserve 와 nDetect 차를 새로운 변수로 생성
    dataset['nObserve'] = dataset['nObserve'].apply(np.log1p)
    dataset['d_obs_det'] = dataset['nObserve'] - dataset['nDetect']

    # permutation importance를 사용해서, 사용할 필드 선택
    drop_columns = ['d_obs_det','g_0','diff_airmass_z_airmass_i','u','airmass_g','airmass_z','nDetect','dered_i_rank','diff_airmass_r_airmass_g','dered_r_rank','dered_g_rank','g_rank','airmass_i_rank','airmass_r_rank','airmass_g_rank','airmass_z_rank','dered_u_rank','r_rank','diff_airmass_u_dered_z','u_rank','z_rank','dered_z_rank','airmass_u_rank','diff_airmass_i_airmass_r','i_rank','airmass_r','z']

    # 필요없는 필드 제거
    dataset = dataset.drop(drop_columns, axis=1).copy()

    # 만들어진 변수들을 저장
    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(dataset.columns):
            f.write(f'{col}\n')

    logging.info('saving features')
    save_data(dataset.values[:n_trn,:], y, train_feature_file)
    save_data(dataset.values[n_trn:,:], None,test_feature_file)

    logging.info('saving target label')
    np.savetxt(target_label_file, y, fmt='%d', delimiter=',')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s     %(levelname)s   %(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--target-label-file', required=True, dest='target_label_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                    args.test_file,
                    args.train_feature_file,
                    args.target_label_file,
                    args.test_feature_file,
                    args.feature_map_file)
    logging.info(f'finished ({time.time() - start:.2f} sec elasped)')

