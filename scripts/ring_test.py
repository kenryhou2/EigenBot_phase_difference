import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge
from matplotlib.transforms import Affine2D
from scipy.stats import circmean, circstd


def plot_circular_histogram(ax, datasets, colors, labels, variable_name):
    """
    Plot concentric rings around the center with mean and standard deviation.
    """
    ring_thickness = 0.15
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        if data is None or len(data) == 0:
            continue

        mean = circmean(data, high=np.pi, low=0)
        std_dev = circstd(data, high=np.pi, low=0)

        start_angle = mean - std_dev
        end_angle = mean + std_dev
        start_angle = max(0, start_angle)
        end_angle = min(np.pi, end_angle)

        outer_radius = 1 - i * ring_thickness
        inner_radius = outer_radius - ring_thickness
        if inner_radius < 0:
            inner_radius = 0
            outer_radius = ring_thickness

        rotation_angle = 3 * np.pi / 2
        print(i, mean, std_dev, start_angle, end_angle, outer_radius, inner_radius)
        wedge = Wedge(
            center=(0, 0),
            r=outer_radius,  # Outer radius
            width=ring_thickness,  # Ring thickness
            theta1=np.degrees(start_angle),
            theta2=np.degrees(end_angle),
            color=color,
            alpha=0.5,
            transform=ax.transData._b,
        )
        transform = Affine2D().rotate(rotation_angle) + ax.transData._b
        wedge.set_transform(transform)
        ax.add_patch(wedge)

        print(
            f"Variable: {variable_name}, Color: {color}, Mean: {mean / np.pi}, STD DEV: {std_dev / np.pi}"
        )

    ax.set_title(variable_name)
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"], fontsize=8)
    ax.set_yticklabels([])

    ax.set_ylim(0, 1)
    ax.set_theta_zero_location("N")  # Move 0 to the North
    ax.set_theta_direction(-1)  # Set the direction of theta to be clockwise
    ax.set_xlim(0, np.pi)


def main(file_paths):
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]
    file_labels = [
        file_path.split("/")[-1].replace(".csv", "") for file_path in file_paths
    ]
    print(file_labels)

    fig = plt.figure(figsize=(8, 8))

    positions = [  # (x0, y0, width, height)
        (0.2, 0.4, 0.25, 0.25),  # Column 2, Top center
    ]
    axs = []
    for pos in positions:
        ax = fig.add_axes(pos, polar=True)
        axs.append(ax)

    for i, (column, ax) in enumerate(zip(dataframes[0].columns, axs)):
        datasets = []
        for df in dataframes:
            if column in df.columns:
                phase_data = (
                    (np.pi * pd.to_numeric(df[column], errors="coerce"))
                    .dropna()
                    .astype(float)
                    .to_numpy()
                )
                datasets.append(phase_data)
            else:
                datasets.append(None)
        print(f"datasets {dataframes[0].columns[i]}: {datasets}")

        colors = ["red", "blue", "orange", "green", "brown", "purple"][: len(datasets)]
        labels = file_labels[: len(datasets)]

        ax = axs[i]
        plot_circular_histogram(ax, datasets, colors, labels, column)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


if __name__ == "__main__":
    file_paths = [
        "data/No_Amputation_Flat.csv",
        "data/Amputate_L2R2.csv",
    ]
    main(file_paths)


def test_ring():
    # Create a polar subplot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Define the ring parameters
    radii = [1, 2, 3, 4, 5, 6]  # Radii for the outer edges of the rings
    thickness = 1  # Thickness of each ring

    colors = [
        "red",
        "blue",
        "orange",
        "green",
        "purple",
        "yellow",
    ]  # Colors for the rings
    # Add rings with thickness to the polar plot
    for i, r in enumerate(radii):
        wedge = Wedge(
            center=(0, 0),
            r=r,
            width=thickness,
            theta1=0,
            theta2=360,
            transform=ax.transData._b,
            facecolor=colors[i],
        )
        ax.add_patch(wedge)

    # Optional: Customize the plot
    ax.set_ylim(
        0, max(radii) - 2
    )  # Set the limits of the radial axis to accommodate thickness
    ax.set_yticks(
        [r + thickness / 2 for r in radii]
    )  # Set radial ticks at the middle of each ring
    ax.set_yticklabels([f"{r} units" for r in radii])  # Optional: Set radial labels

    plt.show()
