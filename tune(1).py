import numpy as np
from lstm_model import LSTMmodel
from experiment_log import log_experiment

HIDDEN_SIZES = [8, 16, 32]
LEARNING_RATES = [0.1, 0.01, 0.001]
EPOCHS_LIST = [5, 10]
BATCH_SIZE = 8
LOG_PATH = "tuning_log.csv"


def grid_search(train_inputs, train_targets, test_inputs, test_targets, sequence_length,
                 hidden_sizes=HIDDEN_SIZES, learning_rates=LEARNING_RATES,
                 epochs_list=EPOCHS_LIST, batch_size=BATCH_SIZE, log_path=LOG_PATH):

    model = LSTMmodel()
    results = []
    total_runs = len(hidden_sizes) * len(learning_rates) * len(epochs_list)
    run_num = 0

    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for epochs in epochs_list:
                run_num += 1
                print(f"\nRun {run_num}/{total_runs}: "
                      f"hidden_size={hidden_size}, lr={learning_rate}, epochs={epochs} ---")

                weights, output_weights, history = model.train_model(
                    train_inputs, train_targets, hidden_size=hidden_size, epochs=epochs,
                    lr=learning_rate, batch_size=batch_size, verbose=False
                )

                test_batches = model.prepare_batches(test_inputs, test_targets, batch_size, shuffle=False)
                test_input_batches = []
                test_target_batches = []
                for batch in test_batches:
                    test_input_batches.append(batch[0])
                    test_target_batches.append(batch[1])
                test_loss, _ = model.evaluate(weights, output_weights, test_input_batches, test_target_batches)

                train_rmse = float(np.sqrt(history[-1]))
                test_rmse = float(np.sqrt(test_loss))

                print(f"train_rmse={train_rmse:.4f}  test_rmse={test_rmse:.4f}")

                log_experiment(
                    log_path,
                    parameters={
                        'model': 'tuning run', 'hidden_size': hidden_size,
                        'epochs': epochs, 'learning_rate': learning_rate,
                        'batch_size': batch_size, 'sequence_length': sequence_length,
                    },
                    results={
                        'train_test_split': f"{len(train_inputs)}:{len(test_inputs)} windows",
                        'dataset_size': len(train_inputs) + len(test_inputs),
                        'train_rmse': round(train_rmse, 4),
                        'test_rmse': round(test_rmse, 4),
                    },
                    notes="hyperparameter grid search",
                )

                results.append({
                    'hidden_size': hidden_size, 'learning_rate': learning_rate, 'epochs': epochs,
                    'train_rmse': train_rmse, 'test_rmse': test_rmse,
                })

    best_result = results[0]
    for result in results:
        if result['test_rmse'] < best_result['test_rmse']:
            best_result = result

    print("\n Grid search complete")
    print(f"Best config: hidden_size={best_result['hidden_size']}, "
          f"lr={best_result['learning_rate']}, epochs={best_result['epochs']} "
          f"= test_rmse={best_result['test_rmse']:.4f}")
    print(f"All {total_runs} runs logged to {log_path}")

    return results, best_result


if __name__ == "__main__":

    np.random.seed(0)
    train_inputs = np.random.randn(120, 24, 10)
    train_targets = np.random.randn(120, 1) * 20 + 80
    test_inputs = np.random.randn(30, 24, 10)
    test_targets = np.random.randn(30, 1) * 20 + 80

    results, best_result = grid_search(
        train_inputs, train_targets, test_inputs, test_targets, sequence_length=24,
        hidden_sizes=[8, 16], learning_rates=[0.1, 0.01], epochs_list=[2],
        log_path="test_tuning_log.csv",
    )
    print("\ntune.py ran successfully.")

    