"""Python code submission file for assignment 1.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the .png images showing your plots for each question are generated.
- 1.a, 1.b, 1.c, 1.d, 2.a, 2.b, 2.c in the codes refer to the corresponding question in the assignment.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def save_and_format_plot(
    filename,
    title=None,
    xlabel=None,
    ylabel=None,
    rotate_xticks=False,
    grid=False,
    legend_title=None,
    figsize=(8, 6),
    is_pie=False,
):
    """
    Saves and formats the current matplotlib plot with consistent styling.

    This function applies common formatting options such as title, axis labels,
    tick rotation, grid, legend, and figure size to the current plot. It then
    saves the plot to the specified .png file.
    """

    plt.gcf().set_size_inches(*figsize)
    if title:
        plt.title(title, fontsize=16)
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    if rotate_xticks:
        plt.xticks(rotation=90, fontsize=12)
    else:
        plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if grid:
        plt.grid(True)
    if legend_title:
        plt.legend(title=legend_title)
    if not is_pie:
        plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def movie_rating_analysis():
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")

    # 1.a Merge Ratings and Movies
    merged_df = pd.merge(ratings_df, movies_df, on="movieId", how="inner")
    merged_df.to_csv("merged_movies_ratings.csv", index=False)

    # === 1.a Top and Bottom Movies by Average Rating ===
    # Start
    movie_avg_ratings = merged_df.groupby("movieId")["rating"].mean()

    # Stop
    movie_avg_ratings = pd.Series(
        movie_avg_ratings).sort_values(ascending=False)
    print("Top 10 Movies by Rating:\n", movie_avg_ratings.head(10))
    print("\nBottom 10 Movies by Rating:\n", movie_avg_ratings.tail(10))

    sns.histplot(merged_df["rating"], kde=True, bins=20)
    save_and_format_plot(
        "distribution_of_movie_ratings.png",
        title="Distribution of Movie Ratings",
        xlabel="Rating",
        ylabel="Frequency",
    )

    # === 1.b Genre Distribution ===
    genres_split = merged_df["genres"].str.split("|", expand=True).stack()
    # Start
    genre_counts = genres_split.value_counts()

    # Stop
    genre_counts = pd.Series(genre_counts).sort_values(ascending=False)
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    save_and_format_plot(
        "movie_genre_distribution.png",
        title="Movie Genre Distribution",
        xlabel="Genre",
        ylabel="Number of Movies",
        rotate_xticks=True,
        figsize=(12, 6),
    )

    # === 1.b Average Rating by Genre ===
    merged_df["genres_split"] = merged_df["genres"].str.split("|")
    merged_df = merged_df.explode("genres_split")
    # Start
    avg_rating_by_genre = merged_df.groupby("genres_split")["rating"].mean()

    # Stop
    avg_rating_by_genre = pd.Series(
        avg_rating_by_genre).sort_values(ascending=False)
    sns.barplot(x=avg_rating_by_genre.index, y=avg_rating_by_genre.values)
    save_and_format_plot(
        "average_rating_by_genre.png",
        title="Average Rating by Genre",
        xlabel="Genre",
        ylabel="Average Rating",
        rotate_xticks=True,
        figsize=(12, 6),
    )

    # === 1.c Average Rating Over Time ===
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], unit="s")
    merged_df["year"] = merged_df["timestamp"].dt.year
    # Start
    avg_rating_by_year = merged_df.groupby("year")["rating"].mean()

    # Stop
    avg_rating_by_year = pd.Series(avg_rating_by_year).sort_index()
    plt.plot(avg_rating_by_year.index, avg_rating_by_year.values, marker="o")
    save_and_format_plot(
        "average_rating_over_time.png",
        title="Average Rating Over Time (Yearly)",
        xlabel="Year",
        ylabel="Average Rating",
        grid=True,
        figsize=(10, 6),
    )

    # === 1.d Ratings Count vs Average Rating per Movie ===
    # Start
    # TO DO, doesn't work still
    movie_rating_sums = merged_df.groupby("movieId")["rating"].sum()
    movie_rating_counts = merged_df.groupby("movieId")["rating"].count()
    # Stop
    movie_stats = pd.DataFrame(
        {
            "movieId": movie_rating_sums.keys(),
            "avg_rating": [
                movie_rating_sums[m] / movie_rating_counts[m]
                for m in movie_rating_sums.keys()
            ],
            "num_ratings": [movie_rating_counts[m] for m in movie_rating_counts.keys()],
        }
    )
    sns.scatterplot(data=movie_stats, x="num_ratings",
                    y="avg_rating", alpha=0.6)
    save_and_format_plot(
        "avg_rating_vs_num_ratings.png",
        title="Average Rating vs. Number of Ratings per Movie",
        xlabel="Number of Ratings",
        ylabel="Average Rating",
        grid=True,
        figsize=(10, 6),
    )


# === Plant Species Analysis ===
def plant_species_analysis():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # --- 2.a Species Counts ---
    # Start
    species_counts = df.groupby("species")["species"].count()

    # Stop
    species_counts = pd.Series(species_counts)
    print("Species Count:\n", species_counts)
    species_counts.plot(kind="bar")
    save_and_format_plot(
        "species_count_bar.png",
        title="Species Count",
        xlabel="Species",
        ylabel="Count",
        figsize=(6, 4),
    )
    species_counts.plot(kind="pie", autopct="%1.1f%%",
                        textprops={"fontsize": 12})
    save_and_format_plot(
        "species_distribution_pie.png",
        title="Species Distribution",
        is_pie=True,
        figsize=(6, 6),
    )

    # --- 2.b Descriptive Stats for Sepal Length ---
    sepal_lengths = df["sepal length (cm)"]
    # Start
    count = sepal_lengths.count()
    mean = sepal_lengths.mean()
    median = sepal_lengths.median()
    min_val = sepal_lengths.min()
    max_val = sepal_lengths.max()
    range_val = max_val - min_val
    std_dev = sepal_lengths.std()
    # Stop
    print("\nDescriptive Statistics for Sepal Length:")
    print("Count:", count)
    print("Mean:", round(mean, 2))
    print("Median:", round(median, 2))
    print("Min:", round(min_val, 2))
    print("Max:", round(max_val, 2))
    print("Range:", round(range_val, 2))
    print("Std Dev:", round(std_dev, 2))

    # create histogram (hint: use the sns.histplot function)
    # Start
    sns.histplot(data=sepal_lengths)
    # Stop
    save_and_format_plot(
        "sepal_length_distribution.png",
        title="Distribution of Sepal Length",
        xlabel="Sepal Length (cm)",
        ylabel="Frequency",
        figsize=(6, 4),
    )

    # create boxplot (hint: use the sns.boxplot function)
    # Start
    sns.boxplot(data=sepal_lengths)
    # Stop

    save_and_format_plot(
        "sepal_length_boxplot.png",
        title="Box Plot of Sepal Length",
        ylabel="Sepal Length (cm)",
        figsize=(4, 6),
    )

    # --- 2.c Petal Length vs Petal Width ---
    species_colors = {
        "setosa": "tab:blue",
        "versicolor": "tab:orange",
        "virginica": "tab:green",
    }

    # create scatter plot
    # Start
    print(df)
    sns.scatterplot(
        data=df,
        x="petal length (cm)",
        y="petal width (cm)",
        hue="species",
        palette=species_colors,
    )

    # Stop
    plt.title("Petal Length vs. Petal Width by Species")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend(title="Species")
    plt.grid(True)
    save_and_format_plot(
        "petal_length_vs_width_scatter.png",
        title="Petal Length vs. Petal Width by Species",
        xlabel="Petal Length (cm)",
        ylabel="Petal Width (cm)",
        grid=True,
        legend_title="Species",
        figsize=(8, 6),
    )


if __name__ == "__main__":
    movie_rating_analysis()
    plant_species_analysis()
