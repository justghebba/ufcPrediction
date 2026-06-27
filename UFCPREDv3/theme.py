def set_ufc_theme():
    """Apply UFC-branded dark theme to all matplotlib/seaborn plots.
    Call once at the start of any script or notebook; every plot downstream
    inherits these settings, ensuring visual consistency across all outputs.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="darkgrid")

    plt.rcParams.update(
        {
            "figure.facecolor": "#111111",
            "axes.facecolor": "#1a1a1a",
            "axes.edgecolor": "#d4a843",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "grid.color": "#333333",
            "grid.alpha": 0.5,
            "lines.color": "#E6242B",
            "patch.edgecolor": "#d4a843",
            "legend.facecolor": "#1a1a1a",
            "legend.edgecolor": "#d4a843",
            "legend.labelcolor": "white",
        }
    )

    sns.set_palette(["#E6242B", "#004B87", "#d4a843", "#888888"])
