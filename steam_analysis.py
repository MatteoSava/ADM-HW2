import matplotlib.pyplot as plt
import pandas as pd

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


def languages_pie(languages):
    """
    RQ4
    This function creates a pie chart with the number of occurrences
    of each language in the dataset
    
    Arguments
        languages : 'language' column of the pandas dataframe
    Returns
        void
    """
        
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.pie(languages.value_counts(), 
        labels = languages.value_counts().index,
        explode = [0.1 if language > 10000 else 0.05 for language in languages.value_counts()],
        autopct = '%.1f%%')
    plt.show()

    
def print_top_languages(top_languages):
    """
    RQ4
    This function prints the top languages
    with the respective number of reviews
    in a nice and readable format
    
    Arguments
        top_languages : list of tuples (e.g [(german, 80), (french, 70), (italian, 60)])
    Returns
        void
    """
    
    print("The three most common languages are:")
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
        print(format_prob(prob_funny), "of the", lang.capitalize(), "reviews were considered 'Funny'")
        
        prob_helpful = compute_prob(f_df_lang, 'votes_helpful', 1)       
        print(format_prob(prob_helpful), "of the", lang.capitalize(), "reviews were considered 'Helpful'")
    
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
