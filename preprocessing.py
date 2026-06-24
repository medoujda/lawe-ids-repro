"""
Preprocessing Module for LAWE-IDS.
Supports: UNSW-NB15, NSL-KDD, CIC-IDS2017, BoT-IoT, CICIoT2023.
Provides load_dataset(), create_splits(), apply_smote().
"""

import numpy as np
import pandas as pd
import warnings
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from config import (UNSW_TRAIN, UNSW_TEST, NSL_KDD_DIR, CICIDS_DIR,
                    TON_IOT_DIR, BOT_IOT_DIR, CICIDS2023_DIR, SEED)

warnings.filterwarnings('ignore')

# NSL-KDD column names (42 total: 40 features + attack_type + difficulty_level)
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

# CIC-IDS2017 CSV filenames
CICIDS_CSV_FILES = [
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-workingHours.pcap_ISCX.csv',
]


def _encode_categoricals(X_df, fit_encoders=None):
    """Label-encode categorical columns. Returns encoded df and encoders dict."""
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    X_out = X_df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        if fit_encoders is None:
            # fit on current data
            X_out[col] = le.fit_transform(X_out[col].astype(str))
        else:
            # use pre-fitted encoder; handle unseen values
            le = fit_encoders[col]
            X_out[col] = X_out[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            X_out[col] = le.transform(X_out[col])
        encoders[col] = le
    return X_out, encoders


def _clean_df(df):
    """Fill NaN and replace Inf in a DataFrame."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    df = df.replace([np.inf, -np.inf], 0)
    return df


def _load_unsw_raw(exclude_attack_categories=None):
    """Load raw UNSW-NB15 train and test DataFrames and return X/y arrays.

    Parameters
    ----------
    exclude_attack_categories : list of str, optional
        Attack categories to exclude from the dataset (e.g., ['Fuzzers']).
        These are removed BEFORE feature processing — flow-based features
        cannot reliably detect payload-dependent attacks like Fuzzers.
    """
    train_df = pd.read_csv(UNSW_TRAIN)
    test_df = pd.read_csv(UNSW_TEST)

    # Optionally exclude specific attack categories (e.g., Fuzzers)
    if exclude_attack_categories:
        for df in (train_df, test_df):
            df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip()
            df.loc[df['attack_cat'] == '', 'attack_cat'] = 'Normal'
        excl = set(exclude_attack_categories)
        n_train_before, n_test_before = len(train_df), len(test_df)
        train_df = train_df[~train_df['attack_cat'].isin(excl)].reset_index(drop=True)
        test_df = test_df[~test_df['attack_cat'].isin(excl)].reset_index(drop=True)
        print(f"[UNSW-NB15] Excluded {sorted(excl)}: "
              f"train {n_train_before}→{len(train_df)}, "
              f"test {n_test_before}→{len(test_df)}")

    drop_cols = ['id', 'attack_cat']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train = train_df.drop('label', axis=1).copy()
    X_test = test_df.drop('label', axis=1).copy()

    # Impute NaN using train statistics
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in numerical_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
    for col in categorical_cols:
        mode_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    # Encode categoricals (fit on train only)
    X_train, encoders = _encode_categoricals(X_train)
    X_test, _ = _encode_categoricals(X_test, fit_encoders=encoders)

    feature_names = X_train.columns.tolist()
    categorical_features = list(encoders.keys())

    return (X_train.values, y_train, X_test.values, y_test,
            feature_names, categorical_features)


def _load_nsl_kdd_raw():
    """Load raw NSL-KDD (train + test combined) and return X/y arrays."""
    train_path = os.path.join(NSL_KDD_DIR, 'KDDTrain+.txt')
    test_path = os.path.join(NSL_KDD_DIR, 'KDDTest+.txt')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"NSL-KDD not found at {train_path}")

    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    # Binary label: normal=0, attack=1
    train_df['label'] = (train_df['attack_type'] != 'normal').astype(int)
    test_df['label'] = (test_df['attack_type'] != 'normal').astype(int)

    # Combine train and test (for random splitting later)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.drop(columns=['attack_type', 'difficulty_level'])

    y = combined['label'].values
    X = combined.drop('label', axis=1).copy()

    X = _clean_df(X)

    # Encode categoricals on combined data
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_names = X.columns.tolist()
    categorical_features = list(encoders.keys())

    return X.values, y, feature_names, categorical_features


def _load_cicids_raw(max_samples=500000):
    """Load raw CIC-IDS2017 (all CSVs combined) and return X/y arrays."""
    dfs = []
    for fname in CICIDS_CSV_FILES:
        fpath = os.path.join(CICIDS_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, encoding='utf-8', low_memory=False)
            df.columns = df.columns.str.strip()
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"CIC-IDS2017 CSVs not found in {CICIDS_DIR}. "
            "Download from: https://www.kaggle.com/datasets/cicdataset/cicids2017"
        )

    data = pd.concat(dfs, ignore_index=True)

    # Find label column
    label_col = 'Label'
    if label_col not in data.columns:
        for c in data.columns:
            if 'label' in c.lower():
                label_col = c
                break

    data['label'] = (data[label_col].str.strip() != 'BENIGN').astype(int)
    data = data.drop(columns=[label_col])

    # Remove metadata columns
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                 'Destination Port', 'Timestamp']
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

    # Handle NaN/Inf
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=1, thresh=int(0.5 * len(data)))
    data = data.dropna()

    # Stratified subsample if needed
    if len(data) > max_samples:
        data, _ = train_test_split(
            data, train_size=max_samples,
            stratify=data['label'], random_state=SEED
        )

    y = data['label'].values
    X = data.drop('label', axis=1).copy()

    # Encode any remaining categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X = X.fillna(0)

    feature_names = X.columns.tolist()
    categorical_features = list(encoders.keys())

    return X.values, y, feature_names, categorical_features


def _load_ton_iot_raw(max_samples=500000):
    """Load CIC-ToN-IoT V2 (parquet, ~4.85M rows → 500K stratified).

    File: ton-iot/CIC-ToN-IoT-V2.parquet
    Features: 77 CICFlowMeter-style flow features.
    Target column: 'Label' (binary 0/1). 'Attack' is the multi-class label.
    Returns (X.values, y, feature_names, categorical_features).
    """
    pq_path = os.path.join(TON_IOT_DIR, 'CIC-ToN-IoT-V2.parquet')
    if not os.path.exists(pq_path):
        raise FileNotFoundError(
            f"TON_IoT not found at {pq_path}. "
            "Expected CIC-ToN-IoT-V2.parquet."
        )

    df = pd.read_parquet(pq_path)

    if 'Label' not in df.columns:
        raise ValueError(
            f"TON_IoT parquet missing 'Label' column. Found: {df.columns.tolist()}"
        )

    # Drop the multi-class target; keep the binary 'Label'
    if 'Attack' in df.columns:
        df = df.drop(columns=['Attack'])

    # Clean NaN / Inf before subsampling so stratify works
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Stratified subsample to max_samples
    if len(df) > max_samples:
        df, _ = train_test_split(
            df, train_size=max_samples,
            stratify=df['Label'], random_state=SEED,
        )

    y = df['Label'].astype(int).values
    X = df.drop(columns=['Label']).copy()

    # Encode any remaining categoricals (typically none after Attack drop)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_names = X.columns.tolist()
    categorical_features = list(encoders.keys())

    return X.values, y, feature_names, categorical_features


def _load_bot_iot_raw(max_samples=500000):
    """Load BoT-IoT 5% 10-best features (single CSV, ~3.67M rows → 500K stratified).

    File: bot-iot/5%/10-best features/UNSW_2018_IoT_Botnet_Final_10_Best.csv
    Separator: ';' (semicolon).
    Target column: 'attack' (binary 0/1).
    Returns (X.values, y, feature_names, categorical_features).
    """
    csv_path = os.path.join(
        BOT_IOT_DIR, '5%', '10-best features',
        'UNSW_2018_IoT_Botnet_Final_10_Best.csv',
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"BoT-IoT not found at {csv_path}. "
            "Download from: https://research.unsw.edu.au/projects/bot-iot-dataset"
        )

    df = pd.read_csv(csv_path, sep=';', low_memory=False)

    if 'attack' not in df.columns:
        raise ValueError(
            f"BoT-IoT CSV missing 'attack' column. Found: {df.columns.tolist()}"
        )

    # Drop identifier/metadata + multi-class label columns + unnamed index column
    drop_cols = ['pkSeqID', 'saddr', 'sport', 'daddr', 'dport',
                 'category', 'subcategory']
    drop_cols += [c for c in df.columns if c.startswith('Unnamed')]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Clean NaN / Inf before subsampling so stratify works
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Stratified subsample to max_samples
    if len(df) > max_samples:
        df, _ = train_test_split(
            df, train_size=max_samples,
            stratify=df['attack'], random_state=SEED,
        )

    y = df['attack'].astype(int).values
    X = df.drop(columns=['attack']).copy()

    # Encode remaining categoricals (typically just 'proto')
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    feature_names = X.columns.tolist()
    categorical_features = list(encoders.keys())

    return X.values, y, feature_names, categorical_features


def _read_cicids2023_split(csv_path, max_samples):
    """Read one CICIoT2023 pre-split CSV and optionally subsample stratified.

    If max_samples is None, the full split is returned (no subsampling).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CICIoT2023 split not found at {csv_path}. "
            "Expected pre-split CSVs in train/, validation/, test/."
        )
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    if 'label' not in df.columns:
        raise ValueError(
            f"{csv_path} missing 'label' column. Found: {df.columns.tolist()}"
        )

    # Clean NaN / Inf before subsample so stratify works
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Binary label: Normal (BenignTraffic) = 0, anything else = 1.
    # Defensive startswith catches both 'BENIGN' and 'BENIGNTRAFFIC' variants.
    lbl = df['label'].astype(str).str.strip().str.upper()
    df['_y'] = (~lbl.str.startswith('BENIGN')).astype(int)
    df = df.drop(columns=['label'])

    if max_samples is not None and len(df) > max_samples:
        df, _ = train_test_split(
            df, train_size=max_samples,
            stratify=df['_y'], random_state=SEED,
        )
    return df


def _load_cicids2023_raw(train_samples=350000, val_samples=50000,
                         test_samples=100000):
    """Load CICIoT2023 from pre-split train/validation/test CSVs.

    Layout expected:
        cicids2023/train/train.csv
        cicids2023/validation/validation.csv
        cicids2023/test/test.csv

    Each CSV has 46 numerical flow features + a 'label' column with strings
    ('BenignTraffic' or an attack name). We stratified-subsample each split
    to keep memory bounded and preserve the provided train/val/test boundary
    (no leakage).

    Returns (X_train, y_train, X_val, y_val, X_test, y_test,
             feature_names, categorical_features).
    """
    train_path = os.path.join(CICIDS2023_DIR, 'train',      'train.csv')
    val_path   = os.path.join(CICIDS2023_DIR, 'validation', 'validation.csv')
    test_path  = os.path.join(CICIDS2023_DIR, 'test',       'test.csv')

    df_train = _read_cicids2023_split(train_path, train_samples)
    df_val   = _read_cicids2023_split(val_path,   val_samples)
    df_test  = _read_cicids2023_split(test_path,  test_samples)

    # Align columns across splits (defensive)
    common_cols = [c for c in df_train.columns
                   if c in df_val.columns and c in df_test.columns]
    df_train = df_train[common_cols]
    df_val   = df_val[common_cols]
    df_test  = df_test[common_cols]

    # Encode categoricals fit on train only (identical treatment in val/test)
    y_train = df_train['_y'].values
    y_val   = df_val['_y'].values
    y_test  = df_test['_y'].values
    X_train = df_train.drop(columns=['_y']).copy()
    X_val   = df_val.drop(columns=['_y']).copy()
    X_test  = df_test.drop(columns=['_y']).copy()

    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Unseen values in val/test → map to the first seen class
        known = set(le.classes_)
        fallback = le.classes_[0]
        X_val[col] = X_val[col].astype(str).apply(
            lambda v: v if v in known else fallback
        )
        X_test[col] = X_test[col].astype(str).apply(
            lambda v: v if v in known else fallback
        )
        X_val[col]  = le.transform(X_val[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    feature_names = X_train.columns.tolist()
    categorical_features = list(encoders.keys())

    return (X_train.values, y_train,
            X_val.values,   y_val,
            X_test.values,  y_test,
            feature_names, categorical_features)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(name, **kwargs):
    """Load a dataset by name.

    Parameters
    ----------
    name : str
        One of 'unsw-nb15', 'nsl-kdd', 'cicids2017'.

    Returns
    -------
    dict with keys:
        X, y, feature_names, n_features, dataset_name, categorical_features
        For UNSW-NB15 (has_official_split=True):
            also X_train_raw, y_train_raw, X_test_raw, y_test_raw
        For NSL-KDD and CIC-IDS2017 (has_official_split=False):
            X and y contain the full combined dataset.
    """
    name_lower = name.lower().replace('_', '-')

    if name_lower == 'unsw-nb15':
        (X_train_raw, y_train_raw, X_test_raw, y_test_raw,
         feature_names, categorical_features) = _load_unsw_raw(**kwargs)

        # Concatenate for consistent API (X, y keys)
        X = np.concatenate([X_train_raw, X_test_raw], axis=0)
        y = np.concatenate([y_train_raw, y_test_raw], axis=0)

        return {
            'X': X,
            'y': y,
            'X_train_raw': X_train_raw,
            'y_train_raw': y_train_raw,
            'X_test_raw': X_test_raw,
            'y_test_raw': y_test_raw,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'dataset_name': 'UNSW-NB15',
            'categorical_features': categorical_features,
            'has_official_split': True,
        }

    elif name_lower == 'nsl-kdd':
        X, y, feature_names, categorical_features = _load_nsl_kdd_raw(**kwargs)
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'dataset_name': 'NSL-KDD',
            'categorical_features': categorical_features,
            'has_official_split': False,
        }

    elif name_lower == 'cicids2017':
        X, y, feature_names, categorical_features = _load_cicids_raw(**kwargs)
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'dataset_name': 'CIC-IDS2017',
            'categorical_features': categorical_features,
            'has_official_split': False,
        }

    elif name_lower == 'ton-iot':
        X, y, feature_names, categorical_features = _load_ton_iot_raw(**kwargs)
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'dataset_name': 'TON_IoT',
            'categorical_features': categorical_features,
            'has_official_split': False,
        }

    elif name_lower == 'bot-iot':
        X, y, feature_names, categorical_features = _load_bot_iot_raw(**kwargs)
        return {
            'X': X,
            'y': y,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'dataset_name': 'BoT-IoT',
            'categorical_features': categorical_features,
            'has_official_split': False,
        }

    elif name_lower == 'cicids2023':
        (X_train_raw, y_train_raw,
         X_val_raw,   y_val_raw,
         X_test_raw,  y_test_raw,
         feature_names, categorical_features) = _load_cicids2023_raw(**kwargs)

        # Concatenate for consistent API (X, y keys)
        X = np.concatenate([X_train_raw, X_val_raw, X_test_raw], axis=0)
        y = np.concatenate([y_train_raw, y_val_raw, y_test_raw], axis=0)

        return {
            'X': X,
            'y': y,
            'X_train_raw': X_train_raw,
            'y_train_raw': y_train_raw,
            'X_val_raw':   X_val_raw,
            'y_val_raw':   y_val_raw,
            'X_test_raw':  X_test_raw,
            'y_test_raw':  y_test_raw,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'dataset_name': 'CICIoT2023',
            'categorical_features': categorical_features,
            'has_official_split': True,
        }

    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. Choose from "
            "'unsw-nb15', 'nsl-kdd', 'cicids2017', "
            "'ton-iot', 'bot-iot', 'cicids2023'."
        )


def create_splits(data, split_type='official', val_ratio=0.2, test_ratio=0.2,
                  random_state=None):
    """Create train/val/test splits with MinMaxScaler fitted on train only.

    Parameters
    ----------
    data : dict
        Output of load_dataset().
    split_type : str
        'official' – use the dataset's pre-defined train/test split (UNSW-NB15).
                     The official train is further split 80/20 into train/val.
        'random'   – random stratified split of all data.
    val_ratio : float
        Fraction of data used for validation (used when split_type='random').
    test_ratio : float
        Fraction of data used for test (used when split_type='random').
    random_state : int or None
        RNG seed; defaults to SEED from config.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    if random_state is None:
        random_state = SEED

    if split_type == 'official' and data.get('has_official_split', False):
        X_tr_raw = data['X_train_raw']
        y_tr_raw = data['y_train_raw']
        X_test = data['X_test_raw']
        y_test = data['y_test_raw']

        if 'X_val_raw' in data and 'y_val_raw' in data:
            # CICIoT2023: use the provided train/val/test split as-is
            X_train, y_train = X_tr_raw, y_tr_raw
            X_val,   y_val   = data['X_val_raw'], data['y_val_raw']
        else:
            # UNSW-NB15: carve val from official train 80/20
            X_train, X_val, y_train, y_val = train_test_split(
                X_tr_raw, y_tr_raw,
                test_size=0.2,
                stratify=y_tr_raw,
                random_state=random_state
            )
    else:
        # Random stratified split for NSL-KDD, CIC-IDS2017 (and fallback)
        X_all = data['X']
        y_all = data['y']

        # First carve out test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all,
            test_size=test_ratio,
            stratify=y_all,
            random_state=random_state
        )

        # Then carve out val from the remainder
        # val_ratio is relative to the original total
        val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            stratify=y_temp,
            random_state=random_state
        )

    # Fit scaler on train only, transform val and test
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'n_features': X_train_scaled.shape[1],
    }


def apply_smote(X, y, random_state=None):
    """Apply SMOTE to balance binary classes.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    random_state : int or None

    Returns
    -------
    X_resampled, y_resampled
    """
    if random_state is None:
        random_state = SEED
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
