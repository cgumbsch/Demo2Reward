import os

import matplotlib.pyplot as plt


def log_ab_plot_tikz(plotname, a, b, a_label="", b_label="",
                     y_label="Reward", x_label="t", mode="train",
                     ylim_min=-0.1, ylim_max=1.1, out_dir="tikz"):
    """Create a two-line plot and export it as TikZ (.tex) and PNG."""
    from matplot2tikz import save as tikz_save

    print(f"Logging ab plot to {plotname}")
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    assert len(a) == len(b)

    ax.plot(a, label=a_label or "a")
    ax.plot(b, label=b_label or "b")
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, f"{mode}_{plotname}.tex")
    tikz_save(tex_path, figure=fig)
    png_path = os.path.join(out_dir, f"{mode}_{plotname}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return tex_path
