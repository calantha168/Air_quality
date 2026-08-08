import numpy as np
import pandas as pd


def make_sequences(data, feature_cols, target_col='PM2.5', sequence_length=24,
                    station_col='raw_station', datetime_col='datetime'):

    X_fe = []
    target_list = []
    timestamps_list = []

    for station, group in data.groupby(station_col):
        group = group.sort_values(datetime_col)
        features = group[feature_cols].to_numpy()
        station_targets = group[target_col].to_numpy()
        station_timestamps = group[datetime_col].to_numpy()

        for t in range(sequence_length, len(group)):
            X_fe.append(features[t - sequence_length:t])
            target_list.append(station_targets[t])
            timestamps_list.append(station_timestamps[t])

    X = np.stack(X_fe)
    y = np.array(target_list)
    timestamps = np.array(timestamps_list)

    print(f"Sequences {X.shape[0]} across "
          f"{data[station_col].nunique()} stations, shape {X.shape}")

    return X, y, timestamps


def split_into_time_blocks(X, y, timestamps, n_blocks=4):

    order = np.argsort(timestamps)
    X_sorted = X[order]
    y_sorted = y[order]

    blocks = []
    boundaries = np.linspace(0, len(X_sorted), n_blocks + 1, dtype=int)

    for i in range(n_blocks):
        start, end = boundaries[i], boundaries[i + 1]
        X_block = X_sorted[start:end]
        y_block = y_sorted[start:end]
        blocks.append((X_block, y_block))
        print(f"Block {i + 1}/{n_blocks}: {len(X_block)} windows "
              f"({timestamps[order][start]} to {timestamps[order][end - 1]})")

    return blocks, (X_sorted, y_sorted)


if __name__ == "__main__":
    np.random.seed(0)
    n_stations, n_hours, n_features = 3, 500, 5

    rows = []
    for s in range(n_stations):
        dates = pd.date_range("2013-03-01", periods=n_hours, freq="h")
        for i, dt in enumerate(dates):
            row = {"station_id": f"station_{s}", "datetime": dt, "PM2.5": np.random.rand() * 100}
            for f in range(n_features):
                row[f"feat_{f}"] = np.random.randn()
            rows.append(row)

    fake_df = pd.DataFrame(rows)
    feature_cols = [f"feat_{f}" for f in range(n_features)]

    X, y, ts = make_sequences(fake_df, feature_cols, sequence_length=24)
    blocks, (X_full, y_full) = split_into_time_blocks(X, y, ts, n_blocks=4)

    print(f"\nFull dataset (baseline): X {X_full.shape}, y {y_full.shape}")
    print("ime_blocks.py functions ran successfully.")