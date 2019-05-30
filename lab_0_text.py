from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import re

sns.set(style="whitegrid")

stop_words = set(stopwords.words("english"))
path_to_png = "images/"
path_to_datasets = "datasets/"


def get_words_list(data):
    return re.sub(r"[^\\a-zA-Z0-9+#\\n]+", " ",
                  re.sub("<[^>]*>", "", " ".join(data))).lower().split(" ")


def filter_by_set(data, by):
    return [e for e in data if e in by]


def plot_by_list(title, df, x, y, png):
    f, ax = plt.subplots(figsize=(8, 10))
    sns.set_color_codes("muted")
    sns.barplot(x=x, y=y, data=df,
                label="count", color="b")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="",
           xlabel=title)
    sns.despine(left=True, bottom=True)
    f.savefig(png)


def main():
    df = pd.read_json(
        path_to_datasets + "/stackoverflow.json")[["title", "body"]].dropna()
    df_lang = pd.read_csv(
        path_to_datasets + "/programming_languages.csv", sep="\n")

    title_list = get_words_list(df["title"])
    body_list = get_words_list(df["body"])

    df_lang.Name = df_lang.Name.apply(lambda col: col.lower())
    lang_list = df_lang.Name.tolist()

    filtred_title = filter_by_set(title_list, lang_list)
    filtred_body = filter_by_set(body_list, lang_list)

    df1 = pd.Index(filtred_title).value_counts() \
        .rename_axis("lang") \
        .reset_index(name="count_1")

    df2 = pd.Index(filtred_body).value_counts() \
        .rename_axis("lang") \
        .reset_index(name="count_2")

    cl_df = pd.merge(df1, df2, how="inner", on="lang").head(25)

    plot_by_list(
        "title", df1.head(25),"count_1", "lang", path_to_png + "lang_title.png")

    plot_by_list(
        "body", df2.head(25), "count_2", "lang", path_to_png + "lang_body.png")


if __name__ == "__main__":
    main()
