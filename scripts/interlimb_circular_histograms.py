#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from matplotlib.transforms import Affine2D
from scipy.stats import circmean, circstd


def plot_ring_histogram(ax, datasets, colors, labels, variable_name, vert=True):
    """
    Plot a circular histogram with mean and standard deviation on the provided axes.
    """
    ring_thickness = 0.1
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        if data is None or len(data) == 0:
            print(f"Data is None for {label} for`{variable_name}`")
            continue

        mean = circmean(data, high=np.pi, low=-np.pi)
        std_dev = circstd(data, high=np.pi, low=-np.pi)

        start_angle = mean - std_dev
        end_angle = mean + std_dev
        start_angle = max(0, start_angle)
        end_angle = min(np.pi, end_angle)

        outer_radius = 1 - i * ring_thickness
        inner_radius = outer_radius - ring_thickness
        if inner_radius < 0:
            inner_radius = 0
            outer_radius = ring_thickness

        if vert:
            rotation_angle = 3 * np.pi / 2
        else:
            rotation_angle = 0
        wedge = Wedge(
            center=(0, 0),
            r=outer_radius,  # Outer radius
            width=ring_thickness,  # Ring thickness
            theta1=np.degrees(start_angle),
            theta2=np.degrees(end_angle),
            color=colors[i],
            alpha=0.5,
            transform=ax.transData._b,
        )
        transform = Affine2D().rotate(rotation_angle) + ax.transData._b
        wedge.set_transform(transform)
        ax.add_patch(wedge)

        if vert:
            mean = np.pi - mean # not sure why the mean values were reflected across pi/2 for the vert config

        # Plot the mean as a ray in the same color
        ax.plot(
            [mean, mean], 
            [0, outer_radius],
            color=colors[i],
            linewidth=2,
        )

        # ax.hist(data, bins=30, color=color, alpha=0.5)  # Uncomment to plot the histogram

        # # Annotate the mean and standard deviation
        # ax.annotate(
        #     f"Mean: {mean / np.pi:.2f}π\nStd Dev: {std_dev / np.pi:.2f}π",
        #     xy=(mean, outer_radius),
        #     xytext=(mean + 0.1, outer_radius + 0.1),
        #     fontsize=8,
        #     color=color,
        #     ha="center",
        #     arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=5),
        # )

        print(
            f"Label: {label}, Variable: {variable_name}, Color: {colors[i]}, Mean: {mean / np.pi}, STD DEV: {std_dev / np.pi}"
        )

    # ax.set_title(variable_name)
    # ax.legend(labels, loc="upper right", fontsize=8)
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"], fontsize=20)
    ax.set_yticklabels([])

    ax.set_ylim(0, 1)

    if not vert:
        ax.set_theta_zero_location("E")  # Move 0 to the North
        ax.set_theta_direction(1)  # Set the direction of theta to be clockwise
    else:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
    ax.set_xlim(0, np.pi)


def plot_circular_histogram(ax, datasets, colors, labels, variable_name):
    """
    Plot a circular histogram with mean and standard deviation on the provided axes.
    """
    for data, color, label in zip(datasets, colors, labels):
        if data is None or len(data) == 0:
            continue

        # Plot the mean (histogram) as a ray and the std dev as a filled sector
        mean = circmean(data, high=np.pi, low=0)
        std_dev = circstd(data, high=np.pi, low=0)
        ax.plot(
            [mean, mean],
            [0, max(ax.get_ylim())],
            color=color,
            linewidth=2,
        )
        # ax.hist(
        #     data,
        #     bins=30,
        #     color=color,
        #     alpha=0.5,
        # ) # Uncomment to plot the histogram
        print(f"MEAN: {mean / np.pi} pi, STD DEV: {std_dev / np.pi}")

        theta = np.linspace(max(mean - std_dev, 0), min(mean + std_dev, np.pi), 100)
        r = max(ax.get_ylim()) * np.ones_like(theta)
        ax.fill_between(theta, 0, r, color=color, alpha=0.25)

    print(f"Variable: {variable_name}")
    ax.set_title(variable_name)

    ax.set_xticks([0, np.pi / 2, np.pi])
    labels = [r"$0$, in-phase", r"$\frac{\pi}{2}$", r"$\pi$, anti-phase"]
    ax.set_xticklabels(labels, fontsize=8, ha="left")
    ax.set_rticks([])

    ax.set_ylim(0, 1)
    ax.set_theta_zero_location("E")  # Move 0 to the North
    ax.set_theta_direction(-1)  # Set the direction of theta to be clockwise
    ax.set_xlim(0, np.pi)


def main(file_paths):
    dataframes = [pd.read_csv(file_path) for file_path in file_paths]
    file_labels = [
        file_path.split("/")[-1].replace(".csv", "") for file_path in file_paths
    ]
    print(file_labels)

    # positions = [(0, 1), (1, 1), (2, 1), (0, 0), (1, 0), (0, 2), (1, 2)]
    # fig, axs = plt.subplots(3, 3, subplot_kw={"polar": True}, figsize=(7, 9))
    # axs = axs.flatten()

    fig = plt.figure(figsize=(6, 10))

    # semi_circle_size = 1
    # gs = GridSpec(
    #     6,
    #     3,
    #     height_ratios=[
    #         semi_circle_size,
    #         semi_circle_size,
    #         semi_circle_size,
    #         semi_circle_size,
    #         semi_circle_size,
    #         semi_circle_size,
    #     ],
    #     width_ratios=[1, 1, 1],
    # )
    # axs = [
    #     fig.add_subplot(gs[1, 0], polar=True),
    #     fig.add_subplot(gs[3, 0], polar=True),
    #     fig.add_subplot(gs[0, 1], polar=True),
    #     fig.add_subplot(gs[2, 1], polar=True),
    #     fig.add_subplot(gs[4, 1], polar=True),
    #     fig.add_subplot(gs[1, 2], polar=True),
    #     fig.add_subplot(gs[3, 2], polar=True),
    # ]

    # This is hard code for 7 plots

    # if ifVert:
    positions = [  # (x0, y0, width, height)
        (0.4, 0.7, 0.25, 0.25)  # Column 2, Top center (L1-R1)
        ,(0.4, 0.45, 0.25, 0.25)  # Column 2, Middle center (L2-R2)
        ,(0.4, 0.2, 0.25, 0.25)  # Column 2, Bottom center (L3-R3)
        ,(0.15, 0.6, 0.25, 0.25)  # Column 1, Middle left (L1-L2)
        ,(0.15, 0.35, 0.25, 0.25)  # Column 1, Bottom left (L1-L3)
        ,(0.65, 0.6, 0.25, 0.25)  # Column 3, Middle right (R1-R2
        ,(0.65, 0.35, 0.25, 0.25)  # Column 3, Bottom right (R1-R3)
    ]
    # else:
    #     positions = [  # (x0, y0, width, height)
    #         (0.4, 0.7, 0.25, 0.25),  # Column 2, Top center
    #         (0.15, 0.6, 0.25, 0.25),  # Column 1, Middle left
    #         (0.4, 0.45, 0.25, 0.25),  # Column 2, Middle center
    #         (0.65, 0.6, 0.25, 0.25),  # Column 3, Middle right
    #         (0.15, 0.35, 0.25, 0.25),  # Column 1, Bottom left
    #         (0.4, 0.2, 0.25, 0.25),  # Column 2, Bottom center
    #         (0.65, 0.35, 0.25, 0.25),  # Column 3, Bottom right
    #     ]

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
        # print(f"datasets {dataframes[0].columns[i]}: {datasets}")

        colors = ["red", "blue", "orange", "green", "brown", "purple", "black"][: len(datasets)]
        labels = file_labels[: len(datasets)]

        ax = axs[i]

        if ifRing:
            plot_ring_histogram(ax, datasets, colors, labels, column, ifVert)
            
        else:
            plot_circular_histogram(ax, datasets, colors, labels, column)
        print()

    for ax in axs:
        if not ax.has_data():
            ax.set_visible(False)

    # plt.tight_layout()
    plt.subplots_adjust(wspace=-0.4, hspace=0.4)
    if ifRing:
        plt.savefig("interlimb_ring_histograms.png", dpi=600)
    else:
        plt.savefig("interlimb_circular_histograms.png", dpi=360)

    plt.show()


if __name__ == "__main__":
    ifRing = True
    ifVert = False
    file_paths = [
        "data/No_Amputation_Flat.csv",
        "data/Amputate_L2R2.csv",
        "data/Amputate_R3L3.csv",
        "data/Amputate_L3.csv",
        "data/Amputate_R2.csv",
        "data/Ablation.csv",
        "data/terrain_phase.csv"
    ]
    main(file_paths)
