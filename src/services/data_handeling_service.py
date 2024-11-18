# Function to preprocess user data
import numpy as np

# Service to preprocess the user data
def preprocess_users(users_df):
    users_df['age'] = users_df['age'].replace("NULL", np.nan).astype(float)
    users_df['age'] = users_df['age'].replace(0, users_df['age'].median())
    return users_df


# Service to preprocess book data
def preprocess_books(books_df):
    books_df['year_of_publication'] = books_df['year_of_publication'].replace(0, np.nan)
    books_df['publisher'] = books_df['publisher'].fillna("Unknown Publisher")
    books_df['book_author'] = books_df['book_author'].fillna("Unknown")
    books_df['year_of_publication'] = books_df['year_of_publication'].fillna(
        round(books_df['year_of_publication'].mean())
    )
    books_df['year_of_publication'] = books_df.groupby('book_author')['year_of_publication'].transform(
        lambda x: x.fillna(round(x.mean()))
    )
    return books_df[['isbn', 'book_title', 'book_author', 'year_of_publication', 'publisher']]


# Service to preprocess ratings data
def preprocess_ratings(ratings_df):
    ratings_df['book_rating'] = ratings_df['book_rating'].replace(0, np.nan)
    valid_ratings_df = ratings_df.loc[ratings_df['book_rating'] > 0]

    # Calculate median rating and above-median indicator
    book_median_ratings = valid_ratings_df.groupby('isbn')['book_rating'].median().rename('median_rating').reset_index()
    ratings_df = ratings_df.merge(book_median_ratings, on='isbn')
    ratings_df['above_median'] = (ratings_df['book_rating'] > ratings_df['median_rating']).astype(int)

    # Count the number of reviews
    review_counts = valid_ratings_df.groupby('isbn').size().rename('review_count').reset_index()
    return ratings_df, review_counts


# Service to preprocess genre data
def preprocess_genre(books_genre):
    books_genre['category'] = books_genre['category'].apply(lambda x: x.strip("[]").replace("'", ""))
    return books_genre


# Service to partition the dataset
def partition_books(books_model_df, review_counts, books_genre):
    books_all_info = books_model_df.merge(review_counts, on='isbn', how='left')
    books_all_info = books_all_info.merge(books_genre[['isbn', 'category']], on='isbn', how='left')
    books_all_info['category'] = books_all_info['category'].replace('', np.nan)

    # Partitioning data into three parts based on data completnes
    books_complete_info_df = books_all_info[
        books_all_info['category'].notnull() & books_all_info['review_count'].notnull()]
    books_missing_genre_df = books_all_info[
        books_all_info['category'].isnull() & books_all_info['review_count'].notnull()]
    books_missing_all_df = books_all_info[books_all_info['category'].isnull() & books_all_info['review_count'].isnull()]

    return books_complete_info_df, books_missing_genre_df, books_missing_all_df
