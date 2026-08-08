import numpy as np
from data_loader import load_all_stations
from preprocessing import Preprocessor
from time_blocks import make_sequences, split_into_time_blocks
from lstm_model import LSTMmodel
from experiment_log import log_experiment
from tune import grid_search
from plot_results import (
    plot_baseline_vs_ensemble,
    plot_training_curves,
    plot_tuning_results,
    plot_predictions,
)

SEQUENCE_LENGTH = 24
N_BLOCKS = 4
BATCH_SIZE = 8
LOG_PATH = "experiment_log.csv"
RANDOM_SEED = 42

TUNE_HYPERPARAMS = True
TUNE_SUBSET_SIZE = 2000
TUNE_LOG_PATH = "tuning_log.csv"

FALLBACK_HIDDEN_SIZE = 16
FALLBACK_LEARNING_RATE = 0.01
FALLBACK_EPOCHS = 5


TUNE_HIDDEN_SIZES = [8, 16, 32]
TUNE_LEARNING_RATES = [0.1, 0.01, 0.001]
TUNE_EPOCHS_LIST = [5, 10]

FEATURE_COLS = [
    'PM10', 'SO2', 'NO2', 'CO', 'O3',
    'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM',
]

def _run_parameter(model_name, hidden_size, epochs, learning_rate):
    return {
        'model': model_name,
        'hidden_size': hidden_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': BATCH_SIZE,
        'sequence_length': SEQUENCE_LENGTH,
    }

def get_rmse_and_batches(model, weights, output_weights, inputs, targets):
    batches = model.prepare_batches(inputs, targets, BATCH_SIZE, shuffle=False)
    input_batches = [batch[0] for batch in batches]
    target_batches = [batch[1] for batch in batches]
    avg_loss, _ = model.evaluate(weights, output_weights, input_batches, target_batches)
    return np.sqrt(avg_loss), input_batches, target_batches


def main():
    np.random.seed(RANDOM_SEED)

    raw = load_all_stations(base_url="https://raw.githubusercontent.com/calantha168/Air_quality/main/data")

    processor = Preprocessor()
    processed = processor.fit(raw)
    print(f"Processed shape: {processed.shape}, missing values: {processed.isnull().sum().sum()}")

    inputs, targets, times = make_sequences(
        processed, 
        feature_cols = FEATURE_COLS, 
        target_col = 'PM2.5',
        sequence_length = SEQUENCE_LENGTH
    )
    blocks, (all_inputs, all_targets) = split_into_time_blocks(inputs, targets, times, n_blocks=N_BLOCKS)

    split_point = int(len(all_inputs) * 0.8)
    train_inputs, train_targets = all_inputs[:split_point], all_targets[:split_point]
    test_inputs, test_targets = all_inputs[split_point:], all_targets[split_point:]

    model = LSTMmodel()

    if TUNE_HYPERPARAMS:
        print("\nHyperparameter search (subset)")
        sample_size = min(TUNE_SUBSET_SIZE, len(train_inputs))
        sample_split_point = int(sample_size * 0.8)
        tune_train_inputs, tune_train_targets = train_inputs[:sample_split_point], train_targets[:sample_split_point]
        tune_test_inputs, tune_test_targets = train_inputs[sample_split_point:sample_size], train_targets[sample_split_point:sample_size]

        tuning_results, best_parameter = grid_search(
            tune_train_inputs, tune_train_targets, tune_test_inputs, tune_test_targets,
            sequence_length=SEQUENCE_LENGTH, batch_size=BATCH_SIZE,
            hidden_sizes = TUNE_HIDDEN_SIZES, learning_rates=TUNE_LEARNING_RATES,
            epochs_list = TUNE_EPOCHS_LIST, log_path=TUNE_LOG_PATH,
        )
        hidden_size = best_parameter['hidden_size']
        learning_rate = best_parameter['learning_rate']
        epochs = best_parameter['epochs']
        print(f"\nUsing tuned hyperparameters: hidden_size={hidden_size}, "
              f"learning_rate={learning_rate}, epochs={epochs}")
    else:
        tuning_results = None
        hidden_size = FALLBACK_HIDDEN_SIZE
        learning_rate = FALLBACK_LEARNING_RATE
        epochs = FALLBACK_EPOCHS
        print(f"\nTUNE_HYPERPARAMS=False, using fixed hyperparameters: "
              f"hidden_size={hidden_size}, learning_rate={learning_rate}, epochs={epochs}")

 
    print("\nTraining baseline LSTM (full dataset)")
    baseline_weights, baseline_output_weights, baseline_history = model.train_model(
        train_inputs, train_targets, hidden_size=hidden_size, epochs=epochs,
        lr=learning_rate, batch_size=BATCH_SIZE
    )
    baseline_test_rmse, test_input_batches, test_target_batches = get_rmse_and_batches(
        model, baseline_weights, baseline_output_weights, test_inputs, test_targets
    )
    log_experiment(
        LOG_PATH,
        parameters=_run_parameter('baseline (full dataset)', hidden_size, epochs, learning_rate),
        results={
            'train_test_split': '80:20',
            'dataset_size': len(all_inputs),
            'train_rmse': round(np.sqrt(baseline_history[-1]), 4),
            'test_rmse': round(baseline_test_rmse, 4),
        },
        notes="baseline LSTM, tuned hyperparameters" if TUNE_HYPERPARAMS else "baseline LSTM, fixed hyperparameters",
    )

    print("\nTraining ensemble (4 time-block LSTMs)")
    ensemble_weights, ensemble_output_weights = [], []
    block_histories = []
    for i, (block_inputs, block_targets) in enumerate(blocks):

        block_split_point = int(len(block_inputs) * 0.8)
        print(f"\nBlock {i + 1}/{N_BLOCKS}")
        block_weights, block_output_weights, block_history = model.train_model(
            block_inputs[:block_split_point], block_targets[:block_split_point], hidden_size=hidden_size,
            epochs=epochs, lr=learning_rate, batch_size=BATCH_SIZE
        )
        ensemble_weights.append(block_weights)
        ensemble_output_weights.append(block_output_weights)
        block_histories.append(block_history)

        block_test_rmse, _, _ = get_rmse_and_batches(model, block_weights, block_output_weights, test_inputs, test_targets)
        log_experiment(
            LOG_PATH,
            parameters=_run_parameter(f'ensemble block {i + 1}/{N_BLOCKS}', hidden_size, epochs, learning_rate),
            results={
                'train_test_split': '80:20 (within block)',
                'dataset_size': len(block_inputs),
                'train_rmse': round(np.sqrt(block_history[-1]), 4),
                'test_rmse': round(block_test_rmse, 4),
            },
            notes=f"time block {i + 1} of {N_BLOCKS}, evaluated on the same held-out test set as baseline",
        )


    ensemble_losses = []
    actual_values = []
    predicted_values = []
    for input_batch, target_batch in zip(test_input_batches, test_target_batches):
        y_pred = model.ensemble_predict(ensemble_weights, ensemble_output_weights, input_batch, method='average')
        ensemble_losses.append(model.compute_loss(y_pred, target_batch))
        actual_values.extend(target_batch.flatten())
        predicted_values.extend(y_pred.flatten())
    ensemble_loss = np.mean(ensemble_losses)
    ensemble_test_rmse = np.sqrt(ensemble_loss)

    log_experiment(
        LOG_PATH,
        parameters=_run_parameter('ensemble (combined, average)', hidden_size, epochs, learning_rate),
        results={
            'train_test_split': '80:20',
            'dataset_size': len(all_inputs),
            'test_rmse': round(ensemble_test_rmse, 4),
        },
        notes="4 block models combined via ensemble_predict(method='average')",
    )

    print()
    comparison_summary = model.compute_comparison(
        baseline_metrics={'loss': baseline_test_rmse ** 2},
        ensemble_metrics={'loss': ensemble_loss},
    )
    model.print_comparison(comparison_summary)
    print(f"\nFull experiment log written to {LOG_PATH}")
    if TUNE_HYPERPARAMS:
        print(f"Hyperparameter search log written to {TUNE_LOG_PATH}")

    print("\nSaving plots")
    plot_baseline_vs_ensemble(baseline_test_rmse, ensemble_test_rmse)
    plot_training_curves(baseline_history, block_histories)
    plot_predictions(actual_values, predicted_values)
    if tuning_results is not None:
        plot_tuning_results(tuning_results)


if __name__ == "__main__":
    main()
