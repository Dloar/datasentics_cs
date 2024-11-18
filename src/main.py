import numpy as np
import pandas as pd
from src.handler.get_all_data_handler import GetAllData
from src.handler.models.models_all_missing_handler import ModelAllMissingHandler
from src.handler.models.models_cat_missing_handler import ModelCatMissingHandler
from src.handler.models.models_no_missing_handler import ModelNonMissingHandler
from src.services.data_handeling_service import preprocess_users, preprocess_books, preprocess_ratings, \
    preprocess_genre, partition_books

# Load all data
source_data_all = GetAllData()
books_df = source_data_all.load_books()
ratings_df = source_data_all.load_ratings()
users_df = source_data_all.load_users()
books_genre = source_data_all.genre_books()

# Preprocess data
users_df = preprocess_users(users_df)
books_model_df = preprocess_books(books_df)
ratings_df, review_counts = preprocess_ratings(ratings_df)
books_genre = preprocess_genre(books_genre)

# Partition the data
books_complete_info_df, books_missing_genre_df, books_missing_all_df = partition_books(
    books_model_df, review_counts, books_genre
)

# Model handling
model1_obj = ModelAllMissingHandler(books_missing_both=books_missing_all_df.reset_index())
pred1 = model1_obj.recommend_similar_books(book_isbn='B0002K6K8O')
print(pred1[['isbn', 'book_title', 'book_author']])

model2_obj = ModelCatMissingHandler(books_missing_genre=books_missing_genre_df.reset_index(),
                                    ratings_df=ratings_df)
pred2 = model2_obj.hybrid_recommend_books(user_id=278851)
print(pred2)

model3_obj = ModelNonMissingHandler(books_complete_info=books_complete_info_df.reset_index(),
                                    ratings_df=ratings_df)
pred3 = model3_obj.hybrid_recommend_books(user_id=8)
print(pred3)

breakpoint()