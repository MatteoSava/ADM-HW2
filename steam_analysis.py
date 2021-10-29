import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def plot_insight_graphs(df):
    """
    To gain general insights from the DataFrame df, plot an histogram for each column and an heatmap of the correlations.

    Args:
        df: DataFrame to perform operations on.

    Returns:
        void.

    """

    #plot an histogram for each column
    df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    plt.figure(figsize=(12, 10))

    #print a table of the correlations of the dataset
    corr = df.corr()
    sns.heatmap(corr, 
                cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                annot=True, annot_kws={"size": 8}, square=True)

def describe_dataset(df):
    """
    Print a list of the columns of the DataFrame df, print a description of df.

    Args:
        df: DataFrame to perform operations on.

    Returns:
        void.

    """
    #print columns list
    print(df.info())
    #print dataset description
    print(df.describe())

    #print average playtime at review
    average_playtime = df['author.playtime_at_review'].describe().mean()/60/60/24
    print(f'Average playtime at review: {average_playtime} days')

def print_dataset_sample(df, row_num):
    """
    Plot a sample of row_num of the DataFrame df taken from the top.

    Args:
        df: DataFrame to perform operations on.
        row_num: Number of rows to plot.

    Returns:
        void.

    """
    #print a sample of the dataset
    print(df.head(row_num))

def plot_reviews_chart(df, row_num):
    """
    Plot a bar plot of the number of reviews for each game in descending order.

    Args:
        df: DataFrame to perform operations on.
        row_num: Number of rows to plot.

    Returns:
        void.

    """
    #group dataset by app_name, compute the size of each group, sorted in descending order
    df_group = df.groupby(['app_name']).size().sort_values(ascending=False).head(row_num)
    
    #plot
    f = plt.figure()
    ax = df_group.plot(figsize=(15,6), kind='bar', color = "royalblue")

    #grids
    plt.grid(color = 'lightgray', linestyle='-.')

    #setting labels and the title
    plt.setp(ax, xlabel='games', ylabel='number of reviews', title = 'Reviews for each game')

    #print the graph
    plt.show()

def plot_free_or_purchased(df):
    """
    Plot a table with the number of games purchased vs free, and print relative pie charts.

    Args:
        df: DataFrame to perform operations on.

    Returns:
        void.

    """
    #group dataset by app_name
    df_group = df.groupby(['app_name'])
    #print a table sample with received_for_free and steam_purchase
    print(df_group[['received_for_free', 'steam_purchase']].sum().head(), '\n\n\n')
    
    #print a pie chart of received for free games
    print(f'Games received for free: {df.received_for_free.sum()}/{df.received_for_free.count()}',)
    df.received_for_free.value_counts(normalize=True).plot.pie()
    plt.show()

    #print a pie chart of purchased games
    print(f'Games purchased: {df.steam_purchase.sum()}/{df.steam_purchase.count()}')
    df.steam_purchase.value_counts(normalize=True).plot.pie()
    plt.show()

def best_weighted_vote_score(df):
    """
    Print a list of the best weighted vote score

    Args:
        df: DataFrame to perform operations on.

    Returns:
        void.

    """
    #group dataset by app_name
    df_group = df.groupby(['app_name'])
    #sort value by weighted vote score
    df_weighted_vote = df_group['weighted_vote_score'].mean().sort_values(ascending=False)
    #print top weighted vote score
    print(df_weighted_vote.head())

def recommendations_summary(df):
    """
    Print the five top recommended games and the five least recommended games.

    Args:
        df: DataFrame to perform operations on.

    Returns:
        void.

    """
    #group by app_name
    df_group = df.groupby(['app_name'])
    #sum all recommended group 
    df_recommended = df_group['recommended'].sum().sort_values(ascending=False)
    #print a table of the most recommended games
    print('Most recommended games: ')
    print(df_recommended.head(), '\n\n')
    #print a table of the least recommended games
    print('Least recommended games: ')
    print(df_recommended.tail())

def most_common_review_time(df, count):
    """
    Print the most common time an author reviews an application.

    Args:
        df: DataFrame to perform operations on.
        count: Number of rows to print.

    Returns:
        void.

    """
    #convert timestamp_created column to datetime, removing seconds
    reviews_time = pd.to_datetime(df.timestamp_created, unit='s').dt.floor('Min')
    reviews_time = reviews_time.dt.time

    #group by review time
    grouping = reviews_time.groupby(reviews_time).size().sort_values(ascending=False).head(count)
    time_of = grouping.index.to_list()

    for i in time_of:
        print('{:02d}:{:02d}'.format(i.hour, i.minute))

def reviews_between_time_intervals(df, list_interval):
    """
    Plot how many reviews are made during a list of time intervals.

    Args:
        df: DataFrame to perform operations on.
        list_interval: List of time intervals.

    Returns:
        void.

    """
    #convert timestamp_created column to datetime
    datex = pd.to_datetime(df.timestamp_created, unit='s')
    datex = pd.DataFrame(datex)
    datex.set_index(pd.DatetimeIndex(datex.timestamp_created), inplace=True)

    calc = 0
    value_counts = pd.DataFrame()
    #loop over interval list
    for interval in list_interval:
        start, end = interval
        tot = datex.between_time(start, end).count()
        tot = int(tot[0])
        RES = pd.DataFrame({'time interval (start, end)' : [interval], 'total reviews' : tot})
        value_counts = value_counts.append(RES)
        calc += tot

    value_counts.reset_index(drop = True, inplace = True)
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