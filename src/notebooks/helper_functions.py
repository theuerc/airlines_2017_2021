""" Helper functions for the notebooks. """

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import pickle
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import pandas as pd
from sklearn import datasets
import os
from tqdm import tqdm
from scipy.stats import kstest, norm
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from IPython.core.interactiveshell import InteractiveShell


def get_corr(df):
    """
    Returns a sorted correlation matrix for a dataframe.

    Parameters:
        df (pandas dataframe):
            dataframe to be used

    Returns:
        pandas dataframe:
            sorted correlation matrix (by absolute value)
    """
    f_corr = df.corr(numeric_only=True).unstack()
    # drop self-correlation
    f_corr = f_corr[f_corr != 1]
    # drop duplicates in multiindex
    # https://stackoverflow.com/questions/50223849/drop-duplicates-of-permuted-multi-index
    m = pd.MultiIndex.from_tuples(
        [tuple(x) for x in np.sort(f_corr.index.tolist(), axis=1)]
    ).duplicated()
    f_corr = f_corr[~m].sort_values(ascending=False, key=abs)
    return pd.DataFrame(f_corr.rename("correlations"))


def get_permutation(df):
    """
    gets a permutation of a two column dataframe,
    preserving the index (which could be something like month)
    by shuffling the two values in each row.

    Args:
        df (pd.DataFrame): dataframe with two columns

    Returns:
        float: the difference between the median of the two columns
    """
    df = df.copy().T
    for i in range(len(df.columns)):
        df.iloc[:, i] = np.random.permutation(df.iloc[:, i])
    return (k := df.T.median())[0] - k[1], k[0], k[1]


def get_permutation_list(df, n=10000):
    """
    gets the median difference of many different permutations
    of a two column dataframe

    Args:
        df (pd.DataFrame): dataframe with two columns
        n (int, optional): number of permutations to create. Defaults to 10000.

    Returns:
        pd.DataFrame: dataframe with the difference between the medians of the two columns
    """
    return pd.DataFrame(
        [get_permutation(df) for i in tqdm(range(n))],
        columns=["diff", "median_1", "median_2"],
    )


def plot_boxplots(
    df,
    cols,
    titles,
    nrows,
    ncols,
    plot_title,
    x_label,
    y_label,
    x_tick_labels,
    y_tick_labels,
    filename="delay_boxplots.png",
):
    """Plot boxplots for the columns in the dataframe.

    Args:
        df (dataframe): a pandas dataframe for different years
        cols (list of str): the columns to plot
        nrows (int): the number of rows in the plot
        ncols (int): the number of columns in the plot
        plot_title (str): the title for the plot
        x_label (str): the label for the x axis
        y_label (str): the label for the y axis
        x_tick_labels (list of str): the tick labels for the x axis
        y_tick_labels (list of str): the tick labels for the y axis
        filename (str): the filename to save the plot to
    """
    # create a figure with the specified grid of subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 30))
    # loop over the columns in del_ct and create a boxplot for each one
    for i, col in enumerate(cols):
        sns.boxplot(
            x="carrier_name",
            y=col,
            data=df,
            ax=axs[i // ncols, i % ncols],
            hue="year",
            dodge=True,
            palette="Set2",
        )
        axs[i // ncols, i % ncols].set_title(col)
        axs[i // ncols, i % ncols].set_xlabel(x_label)
        axs[i // ncols, i % ncols].set_ylabel(y_label) if col != "arr_delay" else axs[
            i // ncols, i % ncols
        ].set_ylabel("average delay (minutes)") #noqa
        axs[i // ncols, i % ncols].set_xticklabels(
            x_tick_labels, rotation=30, ha="right"
        )
        # move the x axis labels to the left
        axs[i // ncols, i % ncols].xaxis.set_label_position("top")
        # make the y axis percentage
        axs[i // ncols, i % ncols].yaxis.set_major_formatter(
            PercentFormatter(1)
        ) if col != "arr_delay" else None #noqa
        # add title
        axs[i // ncols, i % ncols].set_title(titles[i])

    # adjust the layout of subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    # add titles to the subplots
    fig.text(
        0.5, 0.92, plot_title, ha="center", va="center", fontsize=16, fontweight="bold"
    )

    # delete empty subplots
    for i in range(nrows):
        for j in range(ncols):
            if (i * ncols + j) >= len(cols):
                fig.delaxes(axs[i, j])
    # save figure
    plt.savefig(f"../output/reports/figures/{filename}", dpi=300)
    plt.show()


def airline_regression(airline_data, airline_name):
    """
    This function takes a DataFrame containing airline data and an airline name, performs
    a cross-validated linear regression, and returns the model and the average mean squared
    error.
    """
    X = airline_data[["carrier_ct"]]
    y = airline_data["return"]

    # Create a linear regression model
    model = LinearRegression()

    # Perform cross-validation with 5 folds
    scores = cross_val_score(model, X, y, cv=12, scoring="neg_mean_squared_error")

    # Convert to positive mean squared error
    rmse_scores = np.sqrt(-scores)

    # Calculate the average mean squared error and print it
    average_rmse = np.mean(rmse_scores)
    print(f"{airline_name} - Average Root Mean Squared Error: {average_rmse}")

    # Fit the line
    model.fit(X, y)

    return model, average_rmse


def plot_airline_regression(df, year):
    """
    This function takes a DataFrame containing airline data, performs a cross-validated
    linear regression for each airline, and creates a 2x3 subplot with well-labeled
    regression lines for each airline. The Mean Squared Error (MSE) is annotated on each subplot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the airline data.
    year : int
        Year of the data.
    """

    # List of airline names
    airlines = df["carrier_name"].unique()

    # Filter the data for each airline and perform regression
    airline_models = []
    airline_rmses = []
    for airline in airlines:
        airline_data = df[df["carrier_name"] == airline]
        airline_model, average_rmse = airline_regression(airline_data, airline)
        airline_models.append(airline_model)
        airline_rmses.append(average_rmse)

    # Create a 2x3 subplot with regression lines for each airline
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, (airline, model, rmse) in enumerate(
        zip(airlines, airline_models, airline_rmses)
    ):
        airline_data = df[df["carrier_name"] == airline]
        X = airline_data[["carrier_ct"]]
        y = airline_data["return"]

        # Plot the regression line
        axes[i].plot(X, y, "o")
        axes[i].plot(X, model.predict(X))
        axes[i].set_title(airline)
        axes[i].set_xlabel("proportion of carrier related delays")
        axes[i].set_ylabel("monthly stock return")

        # give the graphs a title
        axes[i].set_title(f"{airline} - {year}")

        # Annotate the MSE on the graph
        mse_annotation = f"RMSE: {rmse:.4f}"
        axes[i].annotate(
            mse_annotation,
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            color="red",
        )

    plt.show()


def plot_permutation_test(
    airline_permutations,
    median_returns,
    filename="stock_returns_permutation_test.png",
    side="right",
    titles="Permutation Test for Stock Returns for Airlines Before (2017) and During (2021) the Covid-19 Pandemic",
    label="Permuted Median Differences (2017 minus 2021) in Monthly Stock Returns",
):
    """Plot permutation test results for stock returns.

    Args:
        airline_permutations (list): list of tuples containing airline name and corresponding dataframe
        median_returns (dataframe): a pandas dataframe containing median differences
        filename (str): the filename to save the plot to
        side (str): the side of the plot to draw the red shading on. Either 'right' or 'left'
        titles (str): the title of the plot
        label (str): the label of the x-axis
    """
    # create a 3x2 grid of subplots
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))

    # Create an empty DataFrame to store airline names and p-values
    p_values_df = pd.DataFrame(columns=["Airline", "p-value"])

    for i in range(3):
        for j in range(2):
            if i * 2 + j < len(airline_permutations):
                initial_med_diff = median_returns.iloc[i * 2 + j, 1]
                airline_permutations[i * 2 + j][1][["diff"]].plot(
                    kind="hist", bins=25, ax=ax[i, j], legend=True
                )
                ax[i, j].set_title(airline_permutations[i * 2 + j][0])
                max_x = ax[i, j].get_xlim()[1]
                min_x = ax[i, j].get_xlim()[0]
                ax[i, j].axvspan(
                    initial_med_diff, max_x, facecolor="r", alpha=0.5
                ) if side == "right" else ax[i, j].axvspan(
                    min_x, initial_med_diff, facecolor="r", alpha=0.5
                ) #noqa
                ax[i, j].axvline(
                    initial_med_diff, color="red", linestyle="--", linewidth=2
                )
                p_value = (
                    (
                        airline_permutations[i * 2 + j][1]["diff"] > initial_med_diff
                    ).mean()
                    if side == "right"
                    else (
                        airline_permutations[i * 2 + j][1]["diff"] < initial_med_diff
                    ).mean()
                )
                ax[i, j].annotate(
                    f"p-value: {round(p_value,4)}",
                    xy=(0.7, 0.7),
                    xycoords="axes fraction",
                )
                ax[i, j].get_legend().remove()
                ax[i, j].set_xlabel(label)
                ax[i, j].set_ylabel("count")

                # Append the p-value for the current airline to the DataFrame
                p_values_df = pd.concat(
                    [
                        p_values_df,
                        pd.DataFrame(
                            {
                                "Airline": airline_permutations[i * 2 + j][0],
                                "p-value": p_value,
                            },
                            index=[0],
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                )

    plt.suptitle(titles, fontsize=20, y=1.0005)
    plt.tight_layout()
    plt.savefig(f"../output/reports/figures/{filename}", dpi=300)
    plt.show()
    return p_values_df
