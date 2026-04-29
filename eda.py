import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    from wordcloud import STOPWORDS, WordCloud
    from collections import Counter

    return STOPWORDS, WordCloud, mo, pl, plt, sns


@app.cell
def _(pl):
    df = pl.read_csv(
        "potential-talents - Aspiring human resources - seeking human resources.csv"
    )
    df["id"].n_unique()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # EDA
    """)
    return


@app.cell
def _(df):
    df.glimpse()
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First things we notice are:

    - All 104 candidates have unique ids (no duplicates)
    - `job_title` and `location` includes several categorical variables in each column. Depending on the model's performance later and if we have to use a non embedding approach, we might have to split these into separate columns.
    - Some `location` values appear in non-English (e.g. `Türkiye`, `Kanada`). If `location` becomes a feature then normalization might be necessary.
    - Our target variable: `fit` only has NULL values. We figure out what to do with this later (after we check distributions of the other categorical features).
    - `connection` is stored as string with `+` and a trailing whitespace for connections above 500. It makes more sense to convert the data type into integers and add another column ">500 connections".

    Let's first check the distributions and then decide based on this.
    """)
    return


@app.cell
def _(df, pl):
    df_cleaned = df.with_columns(
        pl.col("connection").str.strip_chars(" +").cast(pl.Int64).alias("connection_int"),
        pl.col("connection").str.contains("+", literal=True).alias("over_500_connections"),
    )
    return (df_cleaned,)


@app.cell
def _(df_cleaned, pl, plt, sns):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df_cleaned["connection_int"], ax=axes[0])
    axes[0].set_title("All candidates")

    sns.histplot(
        df_cleaned.filter(~pl.col("over_500_connections"))["connection_int"], ax=axes[1]
    )

    axes[1].set_title("Excluding 500+ connections")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From our histograms, we see that we have a bimodal distribution with essentially 3 clusters:

    - Cluster 1 (0-100): ~48% of candidates (~50 candidates). Heavily right skewed within this range.
    - Cluster 2 (100-400): Only ~7 candidates are here. Very sparse.
    - Cluster 3 (400+): ~47% of candidates (~49 candidates). Mostly dominated by the 500+ connection candidates.

    Now let's look at `job_title`.

    The company is interested in getting accurate searches from keywords “Aspiring human resources” or “seeking human resources”.
    """)
    return


@app.cell
def _(STOPWORDS, df_cleaned, pl):
    stop = set(STOPWORDS)

    unigrams_job = (
        df_cleaned.select(pl.col("job_title").str.to_lowercase().str.split(" "))
        .explode("job_title")
        .filter(
            ~pl.col("job_title").is_in(list(stop))
            & (pl.col("job_title").str.len_chars() > 1)
        )["job_title"]
        .value_counts()
        .sort("count", descending=True)
        .head(30)
    )

    unigrams_job
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see from the unigram that:

    - HR vocabulary dominates (`human`=63, `resources`=58, `aspiring`=35). Most candidates have at least one of these terms and the keyword search did its job.
    - Self descriptions are common: professional (20) and student (16). Students are a meaningful subgroup that lacks employer/role structure compared to other titles.
    - ~5 candidates appear to be English teachers (`english`, `native`, `teacher`, `epik`, `korea` all cluster at count 5 and are likely the same rows). although english teachers could be transitioning to HR professionals or could be good HR candidates, in terms of search quality, these are not great results. Therefore, we shall filter them out (for this case).
    - Company names (`epik`) appear in raw token counts and will need a deliberate keep/strip decision when building embeddings.
    """)
    return


@app.cell
def _(df_cleaned, pl):
    bigrams_job = (
        df_cleaned.with_row_index("title_idx")
        .with_columns(
            pl.col("job_title")
            .str.to_lowercase()
            .str.extract_all(r"[a-z]+")
            .alias("tokens")
        )
        .explode("tokens")
        .with_columns(pl.col("tokens").shift(-1).over("title_idx").alias("next_token"))
        .filter(pl.col("next_token").is_not_null())
        .select(pl.concat_str(["tokens", "next_token"], separator=" ").alias("bigram"))[
            "bigram"
        ]
        .value_counts()
        .sort("count", descending=True)
        .head(20)
    )

    bigrams_job

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see from the bigram that:

    - `human resources` (63) and `aspiring human` (35) match unigram counts.
    - ~13-16 students have `student at` and `and aspiring` terms.
    - ~7 candidates from C.T. Bauer College of Business.
    """)
    return


@app.cell
def _(df_cleaned):
    df_cleaned["job_title"].str.len_chars().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The job title character length statistics show that:

    - There are 7 to 117 characters
    - The distribution is fairly uniform (mean ~= median)

    No problems here
    """)
    return


@app.cell
def _(df, pl):
    df.filter(pl.col("job_title").str.len_chars() == 7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The minimum job title is `Student`, which we've already seen from our unigram and bigram analysis.
    """)
    return


@app.cell
def _(STOPWORDS, WordCloud, df_cleaned):
    text_job = " ".join(df_cleaned["job_title"].to_list())

    wc_job = WordCloud(
        width=900, height=400, stopwords=STOPWORDS, background_color="white"
    ).generate(text_job)
    return (wc_job,)


@app.cell
def _(plt, wc_job):
    plt.figure(figsize=(12, 5))
    plt.imshow(wc_job, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The wordcloud confirms what we found in our unigram and bigram analysis. That is, HR vocabulary has the largest frequency followed by C.T. Bauer graduates and English teachers.

    Let's move onto the `location` feature.
    """)
    return


@app.cell
def _(df_cleaned):
    df_cleaned["location"].value_counts().sort("count", descending=True).head(20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - `Kanada` is the top location (12 candidates) but it is a non English label. We might need to clean this later.
    - There are two Houston's in the dataset: `Houston, Texas Area` and `Houston, Texas`. This is inconsistent and will be cleaned later if `location` is used as a feature.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
