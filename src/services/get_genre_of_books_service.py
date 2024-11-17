import time
from typing import List

import pandas as pd

from src.services.get_book_genre_service import get_book_info


def get_genre_of_books(books_model_df: pd.DataFrame,
                       books_genre: pd.DataFrame,
                       review_counts_list: List):

    done_list_ids = list(books_genre['isbn'].unique())
    books_filter_genre_df = books_model_df.loc[books_model_df['isbn'].isin(review_counts_list)]
    books_filter_genre_df = books_filter_genre_df.loc[~books_filter_genre_df['isbn'].isin(done_list_ids)]
    print(books_filter_genre_df.shape)

    books_filter_genre_df['categories'] = None
    books_filter_genre_df['description'] = None
    i = 0
    for index, row in books_filter_genre_df.iterrows():
        isbn = row['isbn']
        title = row['book_title']
        author = row['book_author']

        # Get book info from Google Books API
        book_info = get_book_info(title, author)
        # Check if book_info is not None and is a dictionary
        if isinstance(book_info, dict):
            books_filter_genre_df.at[index, 'categories'] = book_info['categories']
            books_filter_genre_df.at[index, 'description'] = book_info['description']

            # Print progress only if both categories and description are not None or empty
            if book_info['categories'] and book_info['description']:
                print(f"{isbn}; {book_info['categories']}; {book_info['description'][:200]}")

        # Optional: Add a delay to avoid hitting rate limits
        time.sleep(.1)
        if i % 5000 == 0:
            books_filter_genre_df.to_csv(f'books_filter_genre_df_{i}.csv', index=False)
        i += 1
        print(i)

    books_filter_genre_df[['isbn', 'categories', 'description']].to_csv('file_1.csv')
    return books_filter_genre_df