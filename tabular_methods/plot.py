
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

# Central dictionary for metric properties
METRIC_PROPERTIES = {
    # Evaluation Metrics
    'mean_eval_G0': {
        'title': 'Mean Discounted Return (G0)',
        'ylabel': 'Mean Return'
    },
    'mean_eval_sum_raw_rewards': {
        'title': 'Mean Sum of Raw Rewards',
        'ylabel': 'Mean Total Reward'
    },
    'mean_eval_episode_length': {
        'title': 'Mean Episode Length',
        'ylabel': 'Mean Steps'
    },
    'eval_best_seen_score': {
        'title': 'Best Seen Score',
        'ylabel': 'Best Score'
    },
    # Training Metrics
    'mean_entropy': {
        'title': 'Mean Policy Entropy (Training)',
        'ylabel': 'Entropy'
    },
    'mean_behavioral_loss': {
        'title': 'Mean Behavioral Agent Loss (Training)',
        'ylabel': 'Loss'
    },
    'mean_target_loss': {
        'title': 'Mean Target Agent Loss (Training)',
        'ylabel': 'Loss'
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
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots the average action selection distribution for each algorithm in a grid layout.
    """
    num_algs = len(results)
    if num_algs == 0:
        return

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        for name, df in results.items():
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(8, 6))
            total_counts = Counter()
            distributions = df['eval_action_distribution'].dropna()
            for action_dict in distributions:
                total_counts.update(action_dict)

            if total_counts:
                actions = sorted(total_counts.keys())
                counts = [total_counts[action] for action in actions]
                sns.barplot(x=actions, y=counts, hue=actions, ax=ax_ind,
                            palette="viridis", legend=False)
                ax_ind.set_xticks(range(len(actions)))
                ax_ind.set_xticklabels(actions)

            ax_ind.set_title(f"Action Distribution: {name}")
            ax_ind.set_xlabel('Action')
            ax_ind.set_ylabel('Total Count (Across All Eval Episodes)')
            plt.tight_layout()

            safe_name = name.replace(" ", "_").replace("=", "").replace(".",
                                                                        "")
            filename = f"{file_root}_action_dist_{safe_name}.png"
            os.makedirs(save_dir, exist_ok=True)
            fig_ind.savefig(os.path.join(save_dir, filename))
            plt.close(fig_ind)

    cols = min(3, num_algs)
    rows = math.ceil(num_algs / cols)
    fig_joint, axes = plt.subplots(
        nrows=rows, ncols=cols,
        figsize=(7 * cols, 5 * rows), squeeze=False
    )
    fig_joint.suptitle(
        'Average Action Selection Distribution During Evaluation', fontsize=16)

    for i, (name, df) in enumerate(results.items()):
        ax = axes[i // cols, i % cols]

        total_counts = Counter()
        distributions = df['eval_action_distribution'].dropna()
        for action_dict in distributions:
            total_counts.update(action_dict)

        if not total_counts:
            ax.set_title(f"{name}\n(No action data available)")
            continue

        actions = sorted(total_counts.keys())
        counts = [total_counts[action] for action in actions]
        sns.barplot(x=actions, y=counts, hue=actions, ax=ax, palette="viridis",
                    legend=False)

        ax.set_title(name)
        ax.set_xlabel('Action')
        ax.set_ylabel('Total Count')
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(actions)

    for i in range(num_algs, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        filename = f"{file_root}_action_dist_JOINT.png"
        os.makedirs(save_dir, exist_ok=True)
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
        x_col: str = 'train_steps',
        save_dir: Optional[str] = None,
        file_root: Optional[str] = None,
        color_map: Optional[Dict[str, tuple]] = None
):
    """
    Plots V(s0) vs G0. Creates one joint plot and saves individual plots.
    """
    if not results:
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in
                     zip(results.keys(), palette)}

    # --- 1. JOINT PLOT (All algorithms on one axis) ---
    fig_joint, ax_joint = plt.subplots(1, 1, figsize=(10, 7))
    fig_joint.suptitle('Value Function Accuracy (All Algorithms)', fontsize=16)

    for name, df in results.items():
        color = color_map[name]
        # Plot G0 with a solid line
        plot_mean_std(df, x_col, 'mean_eval_G0', ax_joint,
                      f'{name} - G0 (Actual)', color)
        # Plot V0 with a dashed line using the same color
        plot_mean_std(df, x_col, 'mean_eval_V0', ax_joint,
                      f'{name} - V(s0) (Predicted)', color)
        ax_joint.lines[-1].set_linestyle('--')

    ax_joint.set_xlabel(x_col)
    ax_joint.set_ylabel('Value / Return')
    ax_joint.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir and file_root:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{file_root}_value_accuracy_JOINT.png"
        os.makedirs(save_dir, exist_ok=True)
        fig_joint.savefig(os.path.join(save_dir, filename))

    plt.show()
    plt.close(fig_joint)

    # --- 2. INDIVIDUAL PLOTS (only if saving) ---
    if save_dir and file_root:
        for name, df in results.items():
            fig_ind, ax_ind = plt.subplots(1, 1, figsize=(8, 6))
            color = color_map[name]

            # Use the consistent color for both lines, different styles
            plot_mean_std(df, x_col, 'mean_eval_G0', ax_ind,
                          'Mean Actual Return (G0)', color)
            plot_mean_std(df, x_col, 'mean_eval_V0', ax_ind,
                          'Mean Predicted Value (V(s0))', color)
            ax_ind.lines[-1].set_linestyle('--')

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
    Plots a selection of key evaluation metrics.
    If a color_map is not provided, a temporary one is generated.
    """
    if metrics_to_plot is None:
        metrics_to_plot = [
            'mean_eval_G0', 'mean_eval_sum_raw_rewards',
            'mean_eval_episode_length', 'eval_best_seen_score'
        ]

    if not metrics_to_plot:
        return

    if color_map is None:
        palette = sns.color_palette("colorblind", len(results))
        color_map = {name: color for name, color in
                     zip(results.keys(), palette)}

    cols = 2
    rows = math.ceil(len(metrics_to_plot) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows),
                             squeeze=False)
    fig.suptitle('Evaluation Metrics vs. Training Steps', fontsize=16)

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
        filename = f"{file_root}_evaluation_metrics.png"
        os.makedirs(save_dir, exist_ok=True)
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