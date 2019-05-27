import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json

from pprint import pprint
from datetime import datetime


def transform_times(dict_string):
    dict_string = dict_string \
        .replace("None", '""').replace("'", '"').replace('u"', '"')
    return json.loads(dict_string)['from']


def transform_tags(list_string):
    return list_string[1:-1].split(',')


def popularity_plot(df, png='images/popularity_plot.png'):
    sns_plot = sns.scatterplot(
        x="date", y="score", hue="popularity", size="popularity", sizes=(5, 20), data=df)
    sns_plot.get_figure().savefig(png)


def time_plot(df, png='images/time_plot.png'):
    date_score = df.groupby(['date', 'score'], as_index=False) \
        .mean().groupby('date')['score'].mean().reset_index(name='score')
    sns.relplot(x="date", y="score", data=date_score).savefig(png)


def scored_plot(df, png='images/score_plot.png'):
    sns_plot = sns.scatterplot(
        x="popularity", y="score", hue="popularity", data=df)
    sns_plot.get_figure().savefig(png)


def get_params(df, name):
    return {
        'name': name,
        'mean': df[name].mean(),
        'median': df[name].median(),
        'max': df[name].max(),
        'rad': df[name].max() - df[name].min(),
        'mode': df[name].mode(),
        'std': df[name].std(),
        'd': df[name].std() ** 2
    }


def main():
    metrics = ['score', 'scored_by',
               'aired', 'name', 'popularity', 'genre']

    df = pd.read_csv('datasets/anime.csv', sep=',')[metrics].dropna()
    df['date'] = pd.DatetimeIndex(
        pd.to_datetime(df['aired'].apply(transform_times))).year

    df['genre'] = df['genre'].apply(transform_tags)
    print(df)
    popularity_plot(df)
    time_plot(df)
    scored_plot(df)
    pprint(get_params(df, 'popularity'))

if __name__ == '__main__':
    main()

"""
date_count = df.groupby('year').size().reset_index(name='count')
    date_count.info()
    date_count_fig = sns.relplot(
        x="year", y="count", kind="line", data=date_count)
    date_count_fig.savefig('2.png')
"""
