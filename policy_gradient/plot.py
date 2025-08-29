
import os
from typing import List, Dict, Optional
from collections import Counter

import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import numpy as np


sns.set_theme(style="darkgrid")

METRIC_PROPERTIES = {
    # == Episodic Evaluation: Soft ==
    'mean_soft_eval_G0': {'title': 'Mean Soft-Eval Return (G0)', 'ylabel': 'Mean Return'},
    'mean_soft_eval_sum_raw_rewards': {'title': 'Mean Soft-Eval Raw Rewards', 'ylabel': 'Mean Reward'},
    'mean_soft_eval_sum_shaped_rewards': {'title': 'Mean Soft-Eval Shaped Rewards', 'ylabel': 'Mean Reward'},
    'mean_soft_eval_episode_length': {'title': 'Mean Soft-Eval Episode Length', 'ylabel': 'Mean Steps'},
    'soft_eval_best_seen_score': {'title': 'Best Seen Soft-Eval Score', 'ylabel': 'Best Score'},

    # == Episodic Evaluation: Hard ==
    'mean_hard_eval_G0': {'title': 'Mean Hard-Eval Return (G0)', 'ylabel': 'Mean Return'},
    'mean_hard_eval_sum_raw_rewards': {'title': 'Mean Hard-Eval Raw Rewards', 'ylabel': 'Mean Reward'},
    'mean_hard_eval_sum_shaped_rewards': {'title': 'Mean Hard-Eval Shaped Rewards', 'ylabel': 'Mean Reward'},
    'mean_hard_eval_episode_length': {'title': 'Mean Hard-Eval Episode Length', 'ylabel': 'Mean Steps'},
    'hard_eval_best_seen_score': {'title': 'Best Seen Hard-Eval Score', 'ylabel': 'Best Score'},

    # == Episodic Training Metrics ==
    'mean_entropy': {'title': 'Mean Policy Entropy (Training)', 'ylabel': 'Entropy'},
    'mean_behavioral_loss': {'title': 'Mean Behavioral Agent Loss (Training)', 'ylabel': 'Loss'},
    'mean_target_loss': {'title': 'Mean Target Agent Loss (Training)', 'ylabel': 'Loss'},
    'mean_raw_reward': {'title': 'Mean Raw Reward (Training)', 'ylabel': 'Mean Reward'},
    'mean_shaped_reward': {'title': 'Mean Shaped Reward (Training)', 'ylabel': 'Mean Reward'},

    # == Continuing Training Metrics ==
    'estimated_avg_reward': {'title': 'Estimated Average Reward (Training)', 'ylabel': 'Average Reward'},
    'agent_loss': {'title': 'Mean Agent Loss (Training)', 'ylabel': 'Loss'},
    'td_error': {'title': 'Mean TD Error (Training)', 'ylabel': 'TD Error'},
    'raw_reward': {'title': 'Raw Reward (Training)', 'ylabel': 'Reward'},
    'shaped_reward': {'title': 'Shaped Reward (Training)', 'ylabel': 'Reward'},

    # == Continuing Evaluation Metrics ==
    'mean_soft_eval_mean_raw_reward': {'title': 'Mean Soft-Eval Avg Raw Reward', 'ylabel': 'Mean Raw Reward'},
    'mean_hard_eval_mean_raw_reward': {'title': 'Mean Hard-Eval Avg Raw Reward', 'ylabel': 'Mean Raw Reward'},
    'mean_soft_eval_mean_shaped_rewards': {'title': 'Mean Soft-Eval Avg Shaped Reward', 'ylabel': 'Mean Shaped Reward'},
    'mean_hard_eval_mean_shaped_rewards': {'title': 'Mean Hard-Eval Avg Shaped Reward', 'ylabel': 'Mean Shaped Reward'},
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
    Saves individual plots for each algorithm if a save directory is provided.
    """
    if not results:
        return

    results = {
        name: df
        for name, df in results.items() if not df.empty
    }

    if not results:
        print(
            f"Warning: All experiments provided to "
            f"plot_action_distribution were empty. Skipping plot."
        )
        return

    if eval_types_to_plot is None:
        eval_types_to_plot = ['soft', 'hard']

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
    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        for name, df in results.items():
            fig_ind, axes_ind = plt.subplots(
                nrows=num_rows, ncols=1,
                figsize=(7, 5 * num_rows), squeeze=False
            )
            fig_ind.suptitle(f'Action Distribution: {name}', fontsize=16)

            for row_idx, eval_type in enumerate(eval_types_to_plot):
                ax = axes_ind[row_idx, 0]
                df_filtered = df[df['eval_type'] == eval_type]
                total_counts = Counter()
                distributions = df_filtered[
                    'eval_action_distribution'].dropna()
                for action_dict in distributions:
                    total_counts.update(action_dict)

                if not total_counts:
                    ax.set_title(f"({eval_type.title()} Eval - No Data)")
                    continue

                actions = sorted(total_counts.keys())
                counts = [total_counts[action] for action in actions]
                sns.barplot(x=actions, y=counts, hue=actions, ax=ax,
                            palette="viridis", legend=False)
                ax.set_title(f"{eval_type.title()} Eval")
                ax.set_ylabel('Total Count')
                ax.set_xlabel('Action')
                ax.set_xticks(range(len(actions)))
                ax.set_xticklabels(actions)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Create a safe filename and save the individual plot
            safe_name = "".join(
                c for c in name if c.isalnum() or c in (' ', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')
            filename = f"{file_root}_action_dist_{safe_name}.png"
            fig_ind.savefig(os.path.join(save_dir, filename))
            plt.close(fig_ind)

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

            if row_idx == num_rows - 1:
                ax.set_xlabel('Action')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        filename = f"{file_root}_action_dist_JOINT.png"
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

    results = {
        name: df
        for name, df in results.items() if not df.empty
    }

    if not results:
        print(
            f"Warning: All experiments provided to "
            f"plot_value_accuracy were empty. Skipping plot."
        )
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
    If metrics_to_plot is None, it plots all available evaluation metrics.
    Saves individual metric plots and a joint plot if a save directory is provided.
    """
    if not results:
        print(
            f"Warning: No results dictionary provided to plot_evaluation_metrics. Skipping plot.")
        return

    results = {name: df for name, df in results.items() if not df.empty}
    if not results:
        print(
            f"Warning: All experiments provided to plot_evaluation_metrics were empty. Skipping plot.")
        return

    if metrics_to_plot is None:
        ALL_POSSIBLE_EVAL_METRICS = [
            'mean_soft_eval_G0', 'mean_hard_eval_G0',
            'mean_soft_eval_sum_raw_rewards', 'mean_hard_eval_sum_raw_rewards',
            'mean_soft_eval_sum_shaped_rewards',
            'mean_hard_eval_sum_shaped_rewards',
            'mean_soft_eval_episode_length', 'mean_hard_eval_episode_length',
            'soft_eval_best_seen_score', 'hard_eval_best_seen_score',
            'mean_eval_V0'
        ]
        available_cols = set(next(iter(results.values())).columns)
        metrics_to_plot = [m for m in ALL_POSSIBLE_EVAL_METRICS if
                           m in available_cols]

    if not metrics_to_plot:
        print(
            "Warning: No evaluation metrics found in the provided data to plot.")
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in
                     zip(results.keys(), palette)}

    # NEW LOGIC: Save individual plot for each metric
    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        for metric in metrics_to_plot:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
            plotted = False
            for name, df in results.items():
                if metric in df.columns and not df[metric].isnull().all():
                    plot_mean_std(df=df, x_col=x_col, y_col=metric,
                                  ax=ax_single, label=name,
                                  color=color_map[name])
                    plotted = True

            if plotted:
                props = METRIC_PROPERTIES.get(metric, {
                    'title': metric, 'ylabel': 'Value'
                })
                ax_single.set_title(props['title'])
                ax_single.set_ylabel(props['ylabel'])
                ax_single.set_xlabel(x_col)
                ax_single.legend()
                plt.tight_layout()

                filename = f"{file_root}_{metric}.png"
                fig_single.savefig(os.path.join(save_dir, filename))

            plt.close(fig_single)

    # --- Joint Plot ---
    cols = 2
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             squeeze=False)
    fig.suptitle('Evaluation Metrics vs. Training Steps', fontsize=16)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // cols, i % cols]
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

    for i in range(len(metrics_to_plot), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        # NEW LOGIC: Added os.makedirs here for robustness
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_evaluation_metrics_JOINT.png"
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
    If metrics_to_plot is None, it plots all available training metrics.
    Saves individual metric plots and a joint plot if a save directory is provided.
    """
    if not results:
        print(f"Warning: No results dictionary provided to"
              f" plot_training_metrics. Skipping plot.")
        return

    results = {name: df for name, df in results.items() if not df.empty}
    if not results:
        print(f"Warning: All experiments provided to "
              f"plot_training_metrics were empty. Skipping plot.")
        return

    if metrics_to_plot is None:
        ALL_POSSIBLE_TRAINING_METRICS = [
            'mean_raw_reward',
            'mean_shaped_reward',
            'mean_entropy',
            'mean_behavioral_loss',
            'mean_target_loss'
        ]
        available_cols = set(next(iter(results.values())).columns)
        metrics_to_plot = [m for m in ALL_POSSIBLE_TRAINING_METRICS if m in available_cols]

    if not metrics_to_plot:
        print("Warning: No training metrics found in the provided data to plot.")
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in zip(results.keys(), palette)}

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        for metric in metrics_to_plot:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
            plotted = False
            for name, df in results.items():
                if metric in df.columns and not df[metric].isnull().all():
                    plot_mean_std(df=df, x_col=x_col, y_col=metric,
                                  ax=ax_single, label=name,
                                  color=color_map[name])
                    plotted = True
            if plotted:
                props = METRIC_PROPERTIES.get(metric, {'title': metric, 'ylabel': 'Value'})
                ax_single.set_title(props['title'])
                ax_single.set_ylabel(props['ylabel'])
                ax_single.set_xlabel(x_col)
                ax_single.legend()
                plt.tight_layout()
                filename = f"{file_root}_{metric}.png"
                fig_single.savefig(os.path.join(save_dir, filename))
            plt.close(fig_single)

    cols = min(3, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    fig.suptitle('Training Metrics vs. Training Steps', fontsize=16)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // cols, i % cols]
        plotted = False
        for name, df in results.items():
            if metric in df.columns and not df[metric].isnull().all():
                plot_mean_std(df=df, x_col=x_col, y_col=metric, ax=ax,
                              label=name, color=color_map[name])
                plotted = True
        props = METRIC_PROPERTIES.get(metric, {'title': metric, 'ylabel': 'Value'})
        ax.set_title(props['title'])
        ax.set_ylabel(props['ylabel'])
        ax.set_xlabel(x_col)
        if plotted:
            ax.legend()

    for i in range(len(metrics_to_plot), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_training_metrics_JOINT.png"
        fig.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig)


def plot_continuing_training_metrics(
        results: Dict[str, pd.DataFrame],
        metrics_to_plot: Optional[List[str]] = None,
        x_col: str = 'train_steps',
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots internal training metrics for continuing tasks.
    If metrics_to_plot is None, it plots all available training metrics.
    Saves individual metric plots and a joint plot if a save directory is provided.
    """
    if not results:
        print(f"Warning: No results dictionary provided to "
              f"plot_continuing_training_metrics. Skipping plot.")
        return

    results = {name: df for name, df in results.items() if not df.empty}
    if not results:
        print(f"Warning: All experiments provided to "
              f"plot_continuing_training_metrics were empty. Skipping plot.")
        return

    if metrics_to_plot is None:
        ALL_POSSIBLE_TRAINING_METRICS = [
            'raw_reward',
            'shaped_reward',
            'estimated_avg_reward',
            'agent_loss',
            'td_error',
            'mean_entropy'
        ]
        available_cols = set(next(iter(results.values())).columns)
        metrics_to_plot = [
            m for m in ALL_POSSIBLE_TRAINING_METRICS
            if m in available_cols
        ]

    if not metrics_to_plot:
        print("Warning: No continuing task training metrics found to plot.")
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in zip(results.keys(), palette)}

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        for metric in metrics_to_plot:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
            plotted = False
            for name, df in results.items():
                if metric in df.columns and not df[metric].isnull().all():
                    bin_size = max(1, df[x_col].max() // 500)
                    binned_df = df.copy()
                    binned_df['binned_steps'] = (binned_df[x_col] // bin_size) * bin_size
                    plot_mean_std(
                        df=binned_df,
                        x_col='binned_steps',
                        y_col=metric,
                        ax=ax_single,
                        label=name,
                        color=color_map[name]
                    )

                    plotted = True
            if plotted:
                props = METRIC_PROPERTIES.get(metric, {'title': metric, 'ylabel': 'Value'})
                ax_single.set_title(props['title'])
                ax_single.set_ylabel(props['ylabel'])
                ax_single.set_xlabel(x_col)
                ax_single.legend()
                plt.tight_layout()
                filename = f"{file_root}_{metric}.png"
                fig_single.savefig(os.path.join(save_dir, filename))
            plt.close(fig_single)

    cols = min(2, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    fig.suptitle('Continuing Task Training Metrics vs. Training Steps', fontsize=16)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // cols, i % cols]
        plotted = False
        for name, df in results.items():
            if metric in df.columns and not df[metric].isnull().all():
                bin_size = max(1, df[x_col].max() // 500)
                binned_df = df.copy()
                binned_df['binned_steps'] = (binned_df[x_col] // bin_size) * bin_size
                plot_mean_std(
                    df=binned_df,
                    x_col='binned_steps',
                    y_col=metric,
                    ax=ax,
                    label=name,
                    color=color_map[name]
                )

                plotted = True
        props = METRIC_PROPERTIES.get(
            metric, {'title': metric, 'ylabel': 'Value'}
        )
        ax.set_title(props['title'])
        ax.set_ylabel(props['ylabel'])
        ax.set_xlabel(x_col)
        if plotted:
            ax.legend()

    for i in range(len(metrics_to_plot), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_continuing_training_JOINT.png"
        fig.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig)


def plot_continuing_evaluation_metrics(
        results: Dict[str, pd.DataFrame],
        metrics_to_plot: Optional[List[str]] = None,
        x_col: str = 'train_steps',
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots key performance metrics from evaluation runs for continuing tasks.
    If metrics_to_plot is None, it plots all available evaluation metrics.
    Saves individual metric plots and a joint plot if a save directory is provided.
    """
    if not results:
        print(
            f"Warning: No results dictionary provided to plot_continuing_evaluation_metrics. Skipping plot.")
        return

    results = {name: df for name, df in results.items() if not df.empty}
    if not results:
        print(
            f"Warning: All experiments provided to plot_continuing_evaluation_metrics were empty. Skipping plot.")
        return

    if metrics_to_plot is None:
        ALL_POSSIBLE_EVAL_METRICS = [
            'mean_soft_eval_mean_raw_reward',
            'mean_hard_eval_mean_raw_reward',
            'mean_soft_eval_mean_shaped_rewards',
            'mean_hard_eval_mean_shaped_rewards'
        ]
        available_cols = set(next(iter(results.values())).columns)
        metrics_to_plot = [m for m in ALL_POSSIBLE_EVAL_METRICS if
                           m in available_cols]

    if not metrics_to_plot:
        print("Warning: No continuing task evaluation metrics found to plot.")
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in
                     zip(results.keys(), palette)}

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        for metric in metrics_to_plot:
            fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
            plotted = False
            for name, df in results.items():
                if metric in df.columns and not df[metric].isnull().all():
                    plot_mean_std(df=df, x_col=x_col, y_col=metric,
                                  ax=ax_single, label=name,
                                  color=color_map[name])
                    plotted = True

            if plotted:
                props = METRIC_PROPERTIES.get(metric, {
                    'title': metric, 'ylabel': 'Mean Reward'
                })
                ax_single.set_title(props['title'])
                ax_single.set_ylabel(props['ylabel'])
                ax_single.set_xlabel(x_col)
                ax_single.legend()
                plt.tight_layout()

                filename = f"{file_root}_{metric}.png"
                fig_single.savefig(os.path.join(save_dir, filename))

            plt.close(fig_single)

    # --- Joint Plot ---
    cols = min(2, len(metrics_to_plot))
    rows = math.ceil(len(metrics_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             squeeze=False)
    fig.suptitle('Continuing Task Evaluation Metrics vs. Training Steps',
                 fontsize=16)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // cols, i % cols]
        plotted = False
        for name, df in results.items():
            if metric in df.columns and not df[metric].isnull().all():
                plot_mean_std(df=df, x_col=x_col, y_col=metric, ax=ax,
                              label=name, color=color_map[name])
                plotted = True

        props = METRIC_PROPERTIES.get(metric, {
            'title': metric, 'ylabel': 'Mean Reward'
        })
        ax.set_title(props['title'])
        ax.set_ylabel(props['ylabel'])
        ax.set_xlabel(x_col)
        if plotted:
            ax.legend()

    for i in range(len(metrics_to_plot), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        filename = f"{file_root}_continuing_evaluation_JOINT.png"
        fig.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig)


def plot_continuing_value_accuracy(
        training_results: Dict[str, pd.DataFrame],
        evaluation_results: Dict[str, pd.DataFrame],
        x_col: str = 'train_steps',
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots the accuracy of the critic in a continuing task by comparing its
    internal average reward estimate against the measured average reward from
    greedy evaluations.
    """
    # Filter empty results from both dictionaries
    training_results = {name: df for name, df in training_results.items() if
                        not df.empty}
    evaluation_results = {name: df for name, df in evaluation_results.items()
                          if not df.empty}

    if not training_results or not evaluation_results:
        print(
            "Warning: Missing training or evaluation data for value accuracy plot. Skipping.")
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(training_results))
        color_map = {name: color for name, color in
                     zip(training_results.keys(), palette)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('Critic Accuracy: Estimated vs. Actual Average Reward',
                 fontsize=16)

    prediction_metric = 'estimated_avg_reward'
    actual_metric = 'mean_hard_eval_mean_shaped_rewards'

    for name, df_train in training_results.items():
        color = color_map.get(name)

        # Plot the agent's internal estimate (Prediction)
        if prediction_metric in df_train.columns and not df_train[
            prediction_metric].isnull().all():
            binned_df = df_train.copy()
            bin_size = max(1, binned_df[x_col].max() // 500)
            binned_df['binned_steps'] = (binned_df[x_col] // bin_size) * bin_size

            plot_mean_std(
                df=binned_df,
                x_col='binned_steps',
                y_col=prediction_metric,
                ax=ax,
                label=f"{name} (Estimated)",
                color=color
            )

            ax.lines[-1].set_linestyle('-')

        # Plot the measured evaluation result (Actual)
        if name in evaluation_results:
            df_eval = evaluation_results[name]
            if actual_metric in df_eval.columns and not df_eval[
                actual_metric].isnull().all():

                plot_mean_std(
                    df=df_eval,
                    x_col=x_col,
                    y_col=actual_metric,
                    ax=ax,
                    label=f"{name} (Actual)",
                    color=color
                )

                ax.lines[-1].set_linestyle('--')

    ax.set_title('Agent\'s Internal Estimate vs. Measured Performance')
    ax.set_xlabel(x_col)
    ax.set_ylabel('Average Reward')
    ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_continuing_value_accuracy.png"
        fig.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig)
