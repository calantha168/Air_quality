import csv
import os
from datetime import datetime


def _format_block(data):
    lines = []
    for key, value in data.items():
        lines.append(f"{key} = {value}")
    return "\n".join(lines)


def log_experiment(log_path, parameters, results, notes=""):

    file_exists = os.path.isfile(log_path)

    if file_exists:
        row_count = 0
        with open(log_path, newline='') as file:
            for line in csv.reader(file):
                row_count = row_count + 1
        header_row_count = 1
        experiment_number = row_count - header_row_count + 1
        if experiment_number < 1:
            experiment_number = 1
    else:
        experiment_number = 1

    row = {
        'Experiment Number': experiment_number,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Notes': notes,
        'Parameters Chosen': _format_block(parameters),
        'Results': _format_block(results),
    }

    with open(log_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged experiment #{experiment_number} -> {log_path}")
    return experiment_number


if __name__ == "__main__":
    test_path = "test_experiment_log.csv"
    if os.path.exists(test_path):
        os.remove(test_path)

    log_experiment(
        test_path,
        parameters={
            'model': 'baseline LSTM', 'hidden_size': 16, 'epochs': 5,
            'learning_rate': 0.01, 'batch_size': 8, 'sequence_length': 24,
        },
        results={
            'train_test_split': '80:20', 'dataset_size': 420768,
            'train_rmse': 12.3, 'test_rmse': 14.1,
        },
        notes="baseline, full dataset",
    )
    log_experiment(
        test_path,
        parameters={
            'model': 'ensemble block 1/4', 'hidden_size': 16, 'epochs': 5,
            'learning_rate': 0.01, 'batch_size': 8, 'sequence_length': 24,
        },
        results={
            'train_test_split': '80:20', 'dataset_size': 105192,
            'train_rmse': 10.8, 'test_rmse': 13.5,
        },
        notes="time block 1 of 4",
    )

    with open(test_path) as f:
        print("\n" + f.read())

    os.remove(test_path)
    print("experiment_log.py works correctly.")