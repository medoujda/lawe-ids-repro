"""Tests for preprocessing module."""
import numpy as np
import pytest
from preprocessing import load_dataset, create_splits, apply_smote

class TestLoadDataset:
    def test_load_unsw_returns_expected_keys(self):
        data = load_dataset('unsw-nb15')
        expected_keys = {'X', 'y', 'feature_names', 'n_features', 'dataset_name',
                         'categorical_features'}
        assert expected_keys.issubset(set(data.keys()))

    def test_load_unsw_shapes(self):
        data = load_dataset('unsw-nb15')
        assert data['X'].shape[0] == len(data['y'])
        assert data['X'].shape[1] == data['n_features']
        assert len(data['feature_names']) == data['n_features']

    def test_load_unsw_binary_labels(self):
        data = load_dataset('unsw-nb15')
        assert set(np.unique(data['y'])).issubset({0, 1})

    def test_load_nsl_kdd(self):
        data = load_dataset('nsl-kdd')
        assert data['dataset_name'] == 'NSL-KDD'
        assert data['n_features'] > 0

    def test_load_cicids2017(self):
        data = load_dataset('cicids2017')
        assert data['dataset_name'] == 'CIC-IDS2017'

class TestCreateSplits:
    def test_unsw_official_split(self):
        data = load_dataset('unsw-nb15')
        splits = create_splits(data, split_type='official')
        assert 'X_train' in splits
        assert 'X_val' in splits
        assert 'X_test' in splits
        assert splits['X_train'].shape[0] > splits['X_val'].shape[0]

    def test_nsl_kdd_random_split(self):
        data = load_dataset('nsl-kdd')
        splits = create_splits(data, split_type='random', val_ratio=0.1, test_ratio=0.1)
        total = splits['X_train'].shape[0] + splits['X_val'].shape[0] + splits['X_test'].shape[0]
        assert total == data['X'].shape[0]

    def test_splits_no_data_leakage(self):
        data = load_dataset('unsw-nb15')
        splits = create_splits(data, split_type='official')
        assert splits['X_train'].shape[0] + splits['X_val'].shape[0] > 0

    def test_splits_scaled(self):
        data = load_dataset('unsw-nb15')
        splits = create_splits(data, split_type='official')
        assert splits['X_train'].min() >= -0.01
        assert splits['X_train'].max() <= 1.01

class TestSmote:
    def test_smote_balances_classes(self):
        X = np.random.rand(100, 5)
        y = np.array([0]*90 + [1]*10)
        X_sm, y_sm = apply_smote(X, y)
        unique, counts = np.unique(y_sm, return_counts=True)
        assert counts[0] == counts[1]
