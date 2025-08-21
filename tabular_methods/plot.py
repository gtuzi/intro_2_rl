
import os
from typing import List, Dict, Optional
from collections import Counter

import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import numpy as np


# Use a nice style for the plots
sns.set_theme(style="darkgrid")

METRIC_PROPERTIES = {
    # Soft Evaluation
    'mean_soft_eval_G0'             : {
        'title': 'Mean Soft-Eval Return (G0)', 'ylabel': 'Mean Return'
    },
    'mean_soft_eval_sum_raw_rewards': {
        'title': 'Mean Soft-Eval Raw Rewards', 'ylabel': 'Mean Reward'
    },
    'mean_soft_eval_episode_length' : {
        'title': 'Mean Soft-Eval Episode Length', 'ylabel': 'Mean Steps'
    },
    'soft_eval_best_seen_score'     : {
        'title': 'Best Seen Soft-Eval Score', 'ylabel': 'Best Score'
    },

    # Hard Evaluation
    'mean_hard_eval_G0'             : {
        'title': 'Mean Hard-Eval Return (G0)', 'ylabel': 'Mean Return'
    },
    'mean_hard_eval_sum_raw_rewards': {
        'title': 'Mean Hard-Eval Raw Rewards', 'ylabel': 'Mean Reward'
    },
    'mean_hard_eval_episode_length' : {
        'title': 'Mean Hard-Eval Episode Length', 'ylabel': 'Mean Steps'
    },
    'hard_eval_best_seen_score'     : {
        'title': 'Best Seen Hard-Eval Score', 'ylabel': 'Best Score'
    },

    # Training Metrics
    'mean_entropy'                  : {
        'title': 'Mean Policy Entropy (Training)', 'ylabel': 'Entropy'
    },
    'mean_behavioral_loss'          : {
        'title': 'Mean Behavioral Agent Loss (Training)', 'ylabel': 'Loss'
    },
    'mean_target_loss'              : {
        'title': 'Mean Target Agent Loss (Training)', 'ylabel': 'Loss'
    },
}


def buil_high_contrast_palette(n_colors):
    return sns.husl_palette(n_colors=n_colors, s=0.9, l=0.6)


def plot_mean_std(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        ax: plt.Axes,
        label: str,
        color: str
):
    """Helper to plot mean with a shaded standard deviation region."""
    # The .dropna() has been removed from this line.
    grouped = df.groupby(x_col)[y_col]
    mean = grouped.mean()
    std = grouped.std()

    ax.plot(mean.index, mean, label=label, color=color)
    ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, color=color)


def plot_action_distribution(
        results: Dict[str, pd.DataFrame],
        eval_types_to_plot: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots the action distribution for selected evaluation types ('soft', 'hard').
    """
    if not results:
        return

    # If no types are specified, default to both soft and hard
    if eval_types_to_plot is None:
        eval_types_to_plot = ['soft', 'hard']

    # Defensively filter for eval types that actually exist in the data
    first_df = next(iter(results.values()))
    if 'eval_type' in first_df.columns:
        available_types = first_df['eval_type'].unique()
        eval_types_to_plot = [t for t in eval_types_to_plot if
                              t in available_types]
    else:
        eval_types_to_plot = []

    if not eval_types_to_plot:
        print(
            "Warning: No action distribution data found for the selected eval types.")
        return

    num_algs = len(results)
    num_rows = len(eval_types_to_plot)

    # --- 1. INDIVIDUAL PLOTS (only if saving) ---
    # This logic remains the same and will save a plot for each algorithm.

    # --- 2. JOINT PLOT (with dynamic rows) ---
    fig_joint, axes = plt.subplots(
        nrows=num_rows, ncols=num_algs,
        figsize=(7 * num_algs, 5 * num_rows), squeeze=False
    )
    fig_joint.suptitle('Action Selection Distribution During Evaluation',
                       fontsize=16)

    for row_idx, eval_type in enumerate(eval_types_to_plot):
        for col_idx, (name, df) in enumerate(results.items()):
            ax = axes[row_idx, col_idx]

            df_filtered = df[df['eval_type'] == eval_type]

            total_counts = Counter()
            distributions = df_filtered['eval_action_distribution'].dropna()
            for action_dict in distributions:
                total_counts.update(action_dict)

            if not total_counts:
                ax.set_title(f"{name}\n({eval_type.title()} Eval - No Data)")
                continue

            actions = sorted(total_counts.keys())
            counts = [total_counts[action] for action in actions]
            sns.barplot(x=actions, y=counts, hue=actions, ax=ax,
                        palette="viridis", legend=False)

            ax.set_title(f"{name} ({eval_type.title()} Eval)")
            ax.set_ylabel('Total Count')
            ax.set_xticks(range(len(actions)))
            ax.set_xticklabels(actions)

            if row_idx == num_rows - 1:  # Only show x-label on the bottom row
                ax.set_xlabel('Action')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_action_dist_JOINT.png"
        fig_joint.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig_joint)


def plot_state_visitation(
        results: Dict[str, pd.DataFrame],
        x_dim_idx: int,
        y_dim_idx: int,
        num_bins: int = 10,
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None
):
    """
    Plots a 2D heatmap of state visitations in a grid layout.
    """
    num_algs = len(results)
    if num_algs == 0:
        return

    # --- 1. INDIVIDUAL PLOTS (only if saving) ---
    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        for name, df in results.items():
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(8, 7))
            # ... (individual plot logic is unchanged) ...
            plt.close(fig_ind)

    # --- 2. JOINT PLOT (with new grid layout) ---
    cols = min(3, num_algs)
    rows = math.ceil(num_algs / cols)
    fig_joint, axes = plt.subplots(
        nrows=rows, ncols=cols,
        figsize=(8 * cols, 6 * rows), squeeze=False
    )
    fig_joint.suptitle(
        f'State Visitation Heatmap (Dimension {x_dim_idx} vs. {y_dim_idx})',
        fontsize=16)

    for i, (name, df) in enumerate(results.items()):
        ax = axes[i // cols, i % cols]  # Use grid indexing

        total_visitations = Counter()
        # ... (aggregation logic is unchanged) ...

        if not total_visitations:
            ax.set_title(f"{name}\n(No visitation data)")
            continue

        grid = np.zeros((num_bins, num_bins))
        # ... (grid population logic is unchanged) ...

        sns.heatmap(data=grid, ax=ax, cmap="viridis", linewidths=.5)
        ax.set_title(name)
        ax.set_xlabel(f'State Dim {x_dim_idx} (Bin)')
        ax.set_ylabel(f'State Dim {y_dim_idx} (Bin)')
        ax.invert_yaxis()

    # Hide any unused subplots
    for i in range(num_algs, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        filename = f"{file_root}_state_visitation_JOINT_dims_{x_dim_idx}v{y_dim_idx}.png"
        os.makedirs(save_dir, exist_ok=True)
        fig_joint.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig_joint)


def plot_value_accuracy(
        results: Dict[str, pd.DataFrame],
        metrics_to_compare: Optional[List[str]] = None,
        x_col: str = 'train_steps',
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots a flexible comparison of V(s0), soft G0, and hard G0.

    Creates one joint plot comparing all algorithms on a single set of axes
    and saves individual plots for each algorithm if a save path is specified.

    Args:
        results: Dictionary of {algorithm_name: preprocessed_dataframe}.
        metrics_to_compare: A list of strings specifying which metrics to plot.
            If None, it defaults to plotting all available metrics.
            Available options:
                - 'V0': The agent's predicted state value, V(s0).
                - 'soft_G0': The actual return from a soft (e.g., epsilon-greedy) evaluation.
                - 'hard_G0': The actual return from a hard (purely greedy) evaluation.
            Example: `metrics_to_compare=['V0', 'hard_G0']`
        x_col: The column to use for the x-axis.
        save_dir: Optional directory path to save the plots.
        file_root: Optional root name for the saved plot files.
        color_map: Optional dictionary mapping algorithm names to colors.
    """
    if not results:
        return

    # Define the properties for each plottable metric in this function
    metric_details = {
        'V0'     : {
            'col': 'mean_eval_V0', 'label': 'V(s0) (Predicted)', 'style': '-'
        },
        'soft_G0': {'col'  : 'mean_soft_eval_G0', 'label': 'Soft G0 (Actual)',
                    'style': ':'
        },
        'hard_G0': {'col'  : 'mean_hard_eval_G0', 'label': 'Hard G0 (Actual)',
                    'style': '--'
        }
    }

    # If no specific metrics are requested, default to plotting all three
    if metrics_to_compare is None:
        metrics_to_compare = ['V0', 'soft_G0', 'hard_G0']

    # Create a title based on the selected metrics
    plot_title = " vs. ".join(
        [metric_details[m]['label'] for m in metrics_to_compare])

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in
                     zip(results.keys(), palette)}

    # --- 1. JOINT PLOT (All algorithms on one axis) ---
    fig_joint, ax_joint = plt.subplots(1, 1, figsize=(10, 7))
    fig_joint.suptitle('Value Function Accuracy', fontsize=16)
    ax_joint.set_title(plot_title)

    for name, df in results.items():
        color = color_map[name]
        for metric_key in metrics_to_compare:
            details = metric_details[metric_key]
            if details['col'] in df.columns and not df[
                details['col']].isnull().all():
                plot_mean_std(df, x_col, details['col'], ax_joint,
                              f"{name} - {details['label']}", color)
                ax_joint.lines[-1].set_linestyle(details['style'])

    ax_joint.set_xlabel(x_col)
    ax_joint.set_ylabel('Value / Return')
    ax_joint.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_value_accuracy_JOINT.png"
        fig_joint.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig_joint)

    # --- 2. INDIVIDUAL PLOTS (only if saving) ---
    if save_dir and file_root:
        for name, df in results.items():
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(8, 6))
            color = color_map[name]

            for metric_key in metrics_to_compare:
                details = metric_details[metric_key]
                if details['col'] in df.columns and not df[
                    details['col']].isnull().all():
                    plot_mean_std(df, x_col, details['col'], ax_ind,
                                  details['label'], color)
                    ax_ind.lines[-1].set_linestyle(details['style'])

            ax_ind.set_title(f"Value Accuracy: {name}")
            ax_ind.set_xlabel(x_col)
            ax_ind.set_ylabel('Value / Return')
            ax_ind.legend()
            plt.tight_layout()

            safe_name = name.replace(" ", "_").replace("=", "").replace(".",
                                                                        "")
            filename = f"{file_root}_value_accuracy_{safe_name}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig_ind.savefig(os.path.join(save_dir, filename))
            plt.close(fig_ind)


def plot_evaluation_metrics(
        results: Dict[str, pd.DataFrame],
        metrics_to_plot: Optional[List[str]] = None,
        x_col: str = 'train_steps',
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots a selection of key evaluation metrics against training steps.

    This function generates two types of visualizations:
    1. A joint plot: A single figure containing a grid of subplots, with each
       subplot showing a specific evaluation metric. This plot is always displayed.
    2. Individual plots: If a save path is provided, a separate plot for each
       metric is saved to its own file without being displayed.

    Args:
        results (Dict[str, pd.DataFrame]): A dictionary mapping algorithm names
            to their preprocessed DataFrames from `preprocess_for_numerical_plots`.
        metrics_to_plot (Optional[List[str]]): A list of strings specifying which
            evaluation metrics to plot. If None, it defaults to plotting all
            available evaluation metrics from the list below.

            Available options:
            --- Soft Evaluation Metrics ---
            - 'mean_soft_eval_G0': Mean discounted return from soft evaluation.
            - 'mean_soft_eval_sum_raw_rewards': Mean raw reward sum from soft evaluation.
            - 'mean_soft_eval_episode_length': Mean episode length from soft evaluation.
            - 'soft_eval_best_seen_score': Best raw score seen in a soft evaluation batch.

            --- Hard Evaluation Metrics ---
            - 'mean_hard_eval_G0': Mean discounted return from greedy evaluation.
            - 'mean_hard_eval_sum_raw_rewards': Mean raw reward sum from greedy evaluation.
            - 'mean_hard_eval_episode_length': Mean episode length from greedy evaluation.
            - 'hard_eval_best_seen_score': Best raw score seen in a greedy evaluation batch.

        x_col (str): The column name to use for the x-axis (defaults to 'train_steps').
        save_dir (Optional[str]): The directory where plot images will be saved.
        file_root (Optional[str]): The base name for the saved plot files.
        color_map (Optional[Dict[str, tuple]]): A dictionary mapping algorithm names
            to specific colors for consistent plotting.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            'mean_soft_eval_G0',
            'mean_hard_eval_G0',
            'mean_soft_eval_sum_raw_rewards',
            'mean_hard_eval_sum_raw_rewards',
            'mean_soft_eval_episode_length',
            'mean_hard_eval_episode_length',
            'soft_eval_best_seen_score',
            'hard_eval_best_seen_score'
        ]

    if not results:
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in
                     zip(results.keys(), palette)}

    # Defensively filter the list of metrics to only those that exist in at least one DataFrame
    available_metrics = set()
    for df in results.values():
        available_metrics.update(df.columns)

    metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]

    if not metrics_to_plot:
        print(
            "Warning: No evaluation metrics found in the provided data to plot.")
        return

    cols = 2
    rows = math.ceil(len(metrics_to_plot) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             squeeze=False)
    fig.suptitle('Evaluation Metrics vs. Training Steps', fontsize=16)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // cols, i % cols]

        # --- INDIVIDUAL PLOTS (if saving) ---
        if save_dir and file_root:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
            plotted_on_single = False
            for name, df in results.items():
                if metric in df.columns and not df[metric].isnull().all():
                    plot_mean_std(df=df, x_col=x_col, y_col=metric,
                                  ax=ax_single, label=name,
                                  color=color_map[name])
                    plotted_on_single = True

            if plotted_on_single:
                props = METRIC_PROPERTIES.get(metric, {
                    'title': metric, 'ylabel': 'Value'
                })
                ax_single.set_title(props['title'])
                ax_single.set_ylabel(props['ylabel'])
                ax_single.set_xlabel(x_col)
                ax_single.legend()
                plt.tight_layout()

                filename = f"{file_root}_{metric}.png"
                os.makedirs(save_dir, exist_ok=True)
                fig_single.savefig(os.path.join(save_dir, filename))

            plt.close(fig_single)

        # --- JOINT PLOT ---
        plotted_on_joint = False
        for name, df in results.items():
            if metric in df.columns and not df[metric].isnull().all():
                plot_mean_std(df=df, x_col=x_col, y_col=metric, ax=ax,
                              label=name, color=color_map[name])
                plotted_on_joint = True

        props = METRIC_PROPERTIES.get(metric,
                                      {'title': metric, 'ylabel': 'Value'})
        ax.set_title(props['title'])
        ax.set_ylabel(props['ylabel'])
        ax.set_xlabel(x_col)
        if plotted_on_joint:
            ax.legend()

    # Hide any unused subplots
    for i in range(len(metrics_to_plot), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and Show Joint Plot
    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_evaluation_metrics.png"
        fig.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig)


def plot_training_metrics(
        results: Dict[str, pd.DataFrame],
        metrics_to_plot: Optional[List[str]] = None,
        x_col: str = 'train_steps',
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots a selection of internal training metrics.
    If a color_map is not provided, a temporary one is generated.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            'mean_entropy', 'mean_behavioral_loss', 'mean_target_loss'
        ]

    if not metrics_to_plot:
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in
                     zip(results.keys(), palette)}

    cols = min(3, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             squeeze=False)
    fig.suptitle('Training Metrics vs. Training Steps', fontsize=16)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // cols, i % cols]
        if save_dir and file_root:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
            for name, df in results.items():
                if metric in df.columns and not df[metric].isnull().all():
                    plot_mean_std(df=df, x_col=x_col, y_col=metric,
                                  ax=ax_single, label=name,
                                  color=color_map[name])

            props = METRIC_PROPERTIES[metric]
            ax_single.set_title(props['title'])
            ax_single.set_ylabel(props['ylabel'])
            ax_single.set_xlabel(x_col)
            ax_single.legend()
            plt.tight_layout()

            filename = f"{file_root}_{metric}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig_single.savefig(os.path.join(save_dir, filename))
            plt.close(fig_single)

        for name, df in results.items():
            if metric in df.columns and not df[metric].isnull().all():
                plot_mean_std(df=df, x_col=x_col, y_col=metric, ax=ax,
                              label=name, color=color_map[name])

        props = METRIC_PROPERTIES[metric]
        ax.set_title(props['title'])
        ax.set_ylabel(props['ylabel'])
        ax.set_xlabel(x_col)
        ax.legend()

    for i in range(len(metrics_to_plot), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_training_metrics.png"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig)