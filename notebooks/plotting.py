import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression


def plot_embeddings_comparison(gt_data,
                               emb_data,
                               val_gt_data=None,
                               val_emb_data=None,
                               num_trials=50):
    """Plot comparison of embeddings and ground truth for train and optionally validation data.

    Args:
        gt_data: Ground truth data batch
        emb_data: Model embeddings
        val_gt_data: Optional validation ground truth data
        val_emb_data: Optional validation embeddings
        num_trials: Number of trials to plot
    """
    num_plots = 4 if val_gt_data is not None else 2
    fig, axes = plt.subplots(1,
                             num_plots,
                             figsize=(7 * num_plots, 6),
                             dpi=300,
                             subplot_kw={'projection': '3d'})

    def process_and_plot(gt_batch, embeddings, ax1, ax2, title_prefix="Train"):
        # Get data for first few trials
        unique_trials = torch.unique(gt_batch.auxilary.trial_id)
        trial_mask = torch.isin(gt_batch.auxilary.trial_id,
                                unique_trials[:num_trials])
        emb_subset = embeddings[trial_mask].detach().cpu().numpy()
        true_subset = gt_batch.latents[trial_mask].detach().cpu().numpy()

        # Align embeddings with ground truth
        lr_model = LinearRegression(fit_intercept=True)
        lr_model.fit(emb_subset, true_subset)
        aligned_embeddings = lr_model.predict(emb_subset)
        r2 = lr_model.score(emb_subset, true_subset)
        print(f"{title_prefix} backward R2: {r2:.3f}")

        # compute forward
        lr_model = LinearRegression(fit_intercept=True)
        lr_model.fit(true_subset, emb_subset)
        #forward_embeddings = lr_model.predict(true_subset)
        r2 = lr_model.score(true_subset, emb_subset)
        print(f"{title_prefix} forward R2: {r2:.3f}")

        # Plot embeddings
        scatter_emb = plot_3d_scatter_and_line(ax1, aligned_embeddings,
                                               f"{title_prefix} Embeddings",
                                               "Embedding")
        fig.colorbar(scatter_emb, ax=ax1, label='Time progression')

        # Plot ground truth
        scatter_gt = plot_3d_scatter_and_line(ax2, true_subset,
                                              f"{title_prefix} Ground Truth",
                                              "Ground Truth")
        fig.colorbar(scatter_gt, ax=ax2, label='Time progression')

    def plot_3d_scatter_and_line(ax, data, title, dim_labels):
        scatter = ax.scatter(data[:, 0],
                             data[:, 1],
                             data[:, 2],
                             c=range(len(data)),
                             cmap='viridis')
        ax.plot(data[:, 0],
                data[:, 1],
                data[:, 2],
                color='gray',
                linewidth=1,
                alpha=0.5)
        ax.set_xlabel(f'{dim_labels} dimension 1')
        ax.set_ylabel(f'{dim_labels} dimension 2')
        ax.set_zlabel(f'{dim_labels} dimension 3')
        ax.set_title(title)
        return scatter

    # Plot training data
    if num_plots == 2:
        process_and_plot(gt_data, emb_data, axes[0], axes[1])
    else:
        process_and_plot(gt_data, emb_data, axes[0], axes[1])
        process_and_plot(val_gt_data, val_emb_data, axes[2], axes[3], "Val")

    plt.tight_layout()
    plt.show()
