import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

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
    

def count_languages(df):
    """
    RQ4
    This function counts how many reviews were written in each of the languages in the dataset
    
    Arguments
        df : pandas dataframe
    Returns
        list of tuples (language, num_of_reviews)
    """
    
    return([(lang, len(frame)) for lang, frame in df.groupby('language')['review_id']])    


def sort_count(count, n = 3, reverse = True):
    """
    RQ4
    This function sorts and slices a list of tuples 
    e.g. [(a, 1), (b, 3), (c, 2)] —> [(b, 3), (c, 2), (a, 1)]
    
    Arguments
        count : list of tuples
    Returns
        sorted and slices list of tuples
    """
    
    top = sorted(count, key = lambda x: x[1], reverse = reverse)
    
    return(top[:n])


def languages_pie(languages, num_of_labels = 9):
    """
    RQ4
    This function creates a pie chart of the number of occurrences
    of each language in the dataset
    
    Arguments
        languages     : 'language' column of the pandas dataframe
        num_of_labels : (int) how many languages are displayed on their own
    Returns
        void
    """
    
    all_langs = languages.value_counts()
    
    shown_langs = all_langs[:num_of_labels]  
    shown_langs['other'] = np.sum(all_langs[num_of_labels:].values)
        
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.pie(
        shown_langs, 
        labels = [lang.capitalize() for lang in shown_langs.index],
        explode = [0.05] + [0] * (shown_langs.index.size - 2) + [0.05],
        autopct = '%.0f%%',
        pctdistance = 0.7,
        startangle = 90,
        textprops = {'fontsize': 15}
        )
    plt.show()

    
def print_top_languages(top_languages):
    """
    RQ4
    This function prints the top languages
    with the respective number of reviews
    in a nice and readable format
    
    Arguments
        top_languages : list of tuples (e.g [('german', 80), ('french', 70), ('italian', 60)])
    Returns
        void
    """
    
    print("The top three most common languages are:\n")
    for lang, num in top_languages:
            print(f"{lang.capitalize()} with {num} reviews")


def filter_by_language(df, languages):
    """
    RQ4
    This function filters the dataframe so it only contains reviews
    written in certain languages, and prints the percentage among these
    of those which were considered 'Funny' 
    and of those which were considered 'Helpful'
    
    Arguments:
        df        : pandas dataframe
        languages : list of languages (e.g ['russian', 'english', 'turkish'])
    Returns:
        f_df      : filtered pandas dataframe
    """
    
    f_df = df[df['language'].isin(languages)]
    
    # this loop prints a report
    # with the percentages of 'Funny' and 'Helpful' reviews
    # for each language in 'languages'
    for lang in languages:
        
        f_df_lang = f_df[f_df['language'] == lang]
        
        prob_funny = compute_prob(f_df_lang, 'votes_funny', 1)       
        print(format_prob(prob_funny, decimals = 0), "of the", lang.capitalize(), "reviews were considered 'Funny'")
        
        prob_helpful = compute_prob(f_df_lang, 'votes_helpful', 1)       
        print(format_prob(prob_helpful, decimals = 0), "of the", lang.capitalize(), "reviews were considered 'Helpful'\n")
    
    return f_df


def compute_prob(df, col_name, x):
    """
    RQ4 RQ7
    This function computes the probability of the values in a column
    being larger than or equal to a certain value
    
    Arguments:
        df        : pandas dataframe
        col_name  : (str) name of the column as defined in the dataframe
        x         : (int or float) number the values in the columns must be larger than or equal to
    Returns:
        prob      : (float) number between 0 and 1
    """
    
    filtered_df = df[df[col_name] >= x]
    
    return filtered_df[col_name].count() / df[col_name].count()


def format_prob(prob, decimals = 2):
    """
    RQ4 RQ7
    This function formats and prints a (float) probability 
    between 0 and 1 as a percentage
    
    Arguments:
        prob      : (float) number between 0 and 1
        decimals  : (int) number of decimals the percentage will be include
    Returns:
        f-string with the probability as a percentage
    """
    
    return f"{100 * prob:.{decimals}f}%"


def ten_popular_reviewers(df):
    """
    RQ5
    This functions finds the top 10 most popular reviewers and their no of reviews
    
    Arguments
        df : pandas dataframe
    Returns
        list of tuples (author.steamid, num_of_reviews)
    """
    
    return(df.groupby(['author.steamid']).size().sort_values(ascending=False).head(10))

def plot_ten_popular_reviewers(df2):
    
    """
    RQ5
    This functions plots the the number of reviews made by the top 10 most popular reviewers
    
    Arguments
        df2 : list of tuples (author.steamid, num_of_reviews)
    Returns
        bar plot of top 10 most popular reviewers
    """
    #plotting the the number of reviews made by the top 10 most popular reviewers
    df2.plot(kind='bar')
    plt.show()
    
def count_and_percent_purchase_free(p,f,l):
    """
    RQ5
    This functions prints the count and percentage of applications purchased, and got for free
    
    Arguments
        p : count of purchased application
        f:  count of free application
        l : length of column
    Returns
         void
    """ 
    a=(p/l)*100
    b=(f/l)*100
    print("Number of applications purchased by the popular reviewer")
    print(f)
    print("Percentage of applications purchased by the popular reviewer" )
    print(b)
    print("Number of applications received for free by the popular reviewer")
    print(p)
    print("Percentage of applications received for free  by the popular reviewer" )
    print(a)    
    
def count_positive_negative_review_free(fn,fp,l2):
    """
    RQ5
    This functions prints the count and percentage of applications purchased, and got for free
    
    Arguments
        fn : count of free application negatively reviewed
        fp:  count of free application positively reviewed
        l2 : length of column
    Returns
         void  """
    a=(fp/l2)*100
    b=(fn/l2)*100
    print("Number of applications received for free and reviewed positively")
    print(fp)
    print("Percentage of applications received for free and reviewed positively" )
    print(a)
    print("Number of applications received for free and reviewed negatively")
    print(fn)
    print("Percentage of applications received for free and reviewed negatively" )
    print(b)  

def count_positive_negative_review_free(pn,pp,l3):
    """
    RQ5
    This functions prints the count and percentage of applications purchased, and got for free
    
    Arguments
        pn : count of purchased application negatively reviewed
        pp:  count of purchased application positively reviewed
        l3 : length of column
    Returns
         void  """
    
    a=(pp/l3)*100
    b=(pn/l3)*100
    print("Number of applications purchased and reviewed positively")
    print(pp)
    print("Percentage of applications purchased and reviewed positively" )
    print(a)
    print("Number of applications purchased and reviewed negatively")
    print(pn)
    print("Percentage of applications purchased  and reviewed negatively" )
    print(b)
    

def avg_time_before_update(df):
    """
    RQ6
    This functions prints the average time (days and minutes) a user lets pass before he updates a review
    
    Arguments
         df : pandas dataframe
        
    Returns
         void
    """
    
    #we extract the column from the dataset
    df4=df['author.playtime_at_review']
    #finding the mean
    t=np.mean(df4)
    #To convert the average time to days and minutes
    t=t/(60*24)
    T=int(t)
    print (T," days",int((t-float(T))*24*60), "Minutes")
    
def removing_nonupdated_reviews(df):
    """
    RQ6
    This functions removes the reviewers who haven't updated the reviews
    
    Arguments
         df : pandas dataframe   
    Returns
         removes new dataframe with reviewer steamid and difference between timestamps of creation and updatiion
    """ 
    
    #We take the difference of the 2 variables ("timestamp created " and "timestamp updated") to find the reviewers who haven't updated the reviews
    difference=df['timestamp_updated']-df['timestamp_created']

    #Storing the values inside a new dataframe Differ
    dataf = {'Differ':difference}
    diff=pd.DataFrame(dataf)
    #creating dataframe of reviewer id
    df6=df['author.steamid']
    df6
    #Concatinating both data frame
    conc = pd.concat([df6, diff], axis=1)
    conc
    
    #Now we delete the reviews which is not updated by removing the reviews with Differ column value zero .
    #So that we get the output dataframe with the time difference between update and
    #creation of reviews and the reviewer id.
    df7 = conc[conc.Differ != 0]
    return(df7)

def count_number_reviews(df):
    """
    RQ6
    This functions counts the number of reviews by each of the top 3 reviewers
    
    Arguments
        df : dataframe
    Returns
        list of tuples (author.steamid, count_of_reviews)
    """
    
    return(df.groupby(['author.steamid']).size().sort_values(ascending=False).head(3))


def plot_top3_reviewers(df2):
    
    """
    RQ6
    This functions plots the top 3 authors that usually update their reviews
    
    Arguments
        df2 : list of tuples (author.steamid, num_of_reviews)
    Returns
        bar plot of top 3 reviewers who usually update their reviews
    """
    #plotting the the number of reviews made by the top 10 most popular reviewers
    df2.plot(kind='bar')
    plt.show()
    

def norm_col(col):
    """
    RQ7 RQ8
    Normalizes a pandas dataframe's column so
    its min value is 0 and its max value is 1
    
    Arguments:
        col : column of a pandas dataframe
    Returns:
        normalized col
    """
    
    col_min = col.min()
    col_max = col.max()
    
    return (col - col_min) / (col_max - col_min)


def gauss(x, mu, sigma):
    """
    RQ8
    Gaussian PDF with parameters mu and sigma
    
    Arguments:
        x     : numpy array of input values
        mu    : mean of the Gaussian
        sigma : standard deviation of the Gaussian
    Returns:
        numpy array
    """
    
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
