import numpy as np
import matplotlib.pyplot as plt

COLOR_BASELINE = "#0072B2"   
COLOR_ENSEMBLE = "#E69F00"   
COLOR_MUTED = "#999999"      
COLOR_ACTUAL = "#333333"     
COLOR_PREDICTED = "#0072B2"  
BLOCK_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"]  


def plot_baseline_vs_ensemble(baseline_rmse, ensemble_rmse, save_path="baseline_vs_ensemble.png"):
    labels = ["Baseline Model", "Ensemble Model"]
    values = [baseline_rmse, ensemble_rmse]
    colors = [COLOR_BASELINE, COLOR_ENSEMBLE]

    figure, axis = plt.subplots(figsize=(6, 5))
    bars = axis.bar(labels, values, color=colors, width=0.5)

    for bar in bars:
        bar_height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar_height,
            f"{bar_height:.2f}", ha="center", va="bottom",
        )

    axis.set_title("Baseline vs Ensemble: Test Error")
    axis.set_ylabel("Error (lower is better)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    print(f"Saved plot: {save_path}")


def plot_training_curves(baseline_history, block_histories, save_path="training_curves.png"):
    figure, axis = plt.subplots(figsize=(7, 5))

    baseline_error = np.sqrt(baseline_history)
    epoch_numbers = range(1, len(baseline_error) + 1)
    axis.plot(epoch_numbers, baseline_error, label="Baseline Model", color="black", linewidth=2)

    block_number = 0
    for block_history in block_histories:
        block_error = np.sqrt(block_history)
        epoch_numbers = range(1, len(block_error) + 1)
        color = BLOCK_COLORS[block_number % len(BLOCK_COLORS)]
        axis.plot(epoch_numbers, block_error, label=f"Block {block_number + 1} Model", color=color, linewidth=2)
        block_number = block_number + 1

    axis.set_title("Training Error Over Time")
    axis.set_xlabel("Epoch Number")
    axis.set_ylabel("Training Error (RMSE)")
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    print(f"Saved plot: {save_path}")


def plot_tuning_results(tuning_results, save_path="tuning_results.png"):
    run_labels = []
    test_errors = []
    run_number = 0
    for result in tuning_results:
        run_number = run_number + 1
        run_labels.append(f"Run {run_number}")
        test_errors.append(result["test_rmse"])

    best_error = min(test_errors)
    bar_colors = []
    for test_error in test_errors:
        if test_error == best_error:
            bar_colors.append(COLOR_ENSEMBLE)
        else:
            bar_colors.append(COLOR_MUTED)

    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(run_labels, test_errors, color=bar_colors)

    for bar, test_error in zip(bars, test_errors):
        if test_error == best_error:
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"best: {test_error:.2f}", ha="center", va="bottom",
            )

    axis.set_title("Hyperparameter Search Results")
    axis.set_xlabel("Setting Tried")
    axis.set_ylabel("Test Error (RMSE)")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    print(f"Saved plot: {save_path}")


def plot_predictions(actual_values, predicted_values, save_path="predictions_vs_actual.png", max_points_to_show=200):
    actual_values = np.array(actual_values).flatten()
    predicted_values = np.array(predicted_values).flatten()

    shown_actual = actual_values[:max_points_to_show]
    shown_predicted = predicted_values[:max_points_to_show]
    time_steps = range(len(shown_actual))

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(time_steps, shown_actual, label="Actual PM2.5", color=COLOR_ACTUAL, linewidth=2)
    axis.plot(time_steps, shown_predicted, label="Predicted PM2.5", color=COLOR_PREDICTED, linewidth=2)

    axis.set_title(f"Predicted vs Actual PM2.5 (first {len(shown_actual)} test points)")
    axis.set_xlabel("Time Step")
    axis.set_ylabel("PM2.5")
    axis.legend()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)
    print(f"Saved plot: {save_path}")