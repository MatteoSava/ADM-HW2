import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def dataset_analysis_table(df):
    df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations

    plt.figure(figsize=(12, 10))
    corr = df.corr()

    sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
                cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                annot=True, annot_kws={"size": 8}, square=True)

def describe_dataset(df):
    print(df.describe())
    print(df.info())
    print(df['author.playtime_at_review'].describe().mean()/60/60/24)

def print_dataset_sample(df, row_num):
    print(df.head(row_num))

def plot_reviews_chart(df):
    df_group = df.groupby(['app_name']).size().sort_values(ascending=False)
    # plotting the DataFrame
    f = plt.figure()
    ax = df_group.plot(figsize=(15,6), kind='bar', color = "royalblue", zorder=3)

    # grids
    plt.grid(color = 'lightgray', linestyle='-.', zorder = 0)

    # setting labels and the title
    plt.setp(ax, xlabel='games', ylabel='number of reviews', title = 'Reviews for each game')

    #print the graph
    plt.show()

def plot_free_or_purchased(df):
    df_group = df.groupby(['app_name'])
    print(df_group[['received_for_free', 'steam_purchase']].sum().head(), '\n\n\n')
    print(f'Games received for free: {df.received_for_free.sum()}/{df.received_for_free.count()}',)
    df.received_for_free.value_counts(normalize=True).plot.pie()
    plt.show()
    print(f'Games purchased: {df.steam_purchase.sum()}/{df.steam_purchase.count()}')
    df.steam_purchase.value_counts(normalize=True).plot.pie()
    plt.show()

def best_weighted_vote_score(df):
    df_group = df.groupby(['app_name'])
    df_weighted_vote = df_group['weighted_vote_score'].mean().sort_values(ascending=False)
    print(df_weighted_vote.head())

def recommendations_summary(df):
    df_group = df.groupby(['app_name'])
    df_recommended = df_group['recommended'].sum().sort_values(ascending=False)
    print('Most recommended games: ')
    print(df_recommended.head(), '\n\n')
    print('Least recommended games: ')
    print(df_recommended.tail())

def most_common_review_time(df, count):
    reviews_time = pd.to_datetime(df.timestamp_created, unit='s').dt.floor('Min')
    reviews_time = reviews_time.dt.time

    grouping = reviews_time.groupby(reviews_time).size().sort_values(ascending=False).head(count)
    time_of = grouping.index.to_list()

    for i in time_of:
        print(f'{i.hour}:{i.minute}')
        #print('{:02d}:{:02d}'.format(i.hour, i.minute))

def reviews_between_time_intervals(df, list_interval):
    datex = pd.to_datetime(df.timestamp_created, unit='s')
    datex = pd.DataFrame(datex)
    datex.set_index(pd.DatetimeIndex(datex.timestamp_created), inplace=True)

    calc = 0
    value_counts = pd.DataFrame()
    for interval in list_interval:
        start, end = interval
        #print(start, end)
        tot = datex.between_time(start, end).count()
        tot = int(tot[0])
        RES = pd.DataFrame({'time interval (start, end)' : [interval], 'total reviews' : tot})
        value_counts = value_counts.append(RES)
        calc += tot
        #print(tot)
    value_counts.reset_index(drop = True, inplace = True)
    #value_counts.drop(['index'])
    #value_counts = value_counts.sort_values(ascending=False)
    print(value_counts)

    # plotting the DataFrame
    f = plt.figure()
    ax = value_counts.plot(figsize=(15,6), kind='bar', color = "royalblue", zorder=3)

    # grids
    plt.grid(color = 'lightgray', linestyle='-.', zorder = 0)

    # setting labels and the title
    plt.setp(ax, xlabel='time intervals', ylabel='number of reviews', title = 'Reviews for each time interval')

    #print the graph
    plt.show()
    #print("total", calc)