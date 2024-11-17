import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader


class ModelNonMissingHandler:
    def __init__(self, books_complete_info, ratings_df):
        self.books_missing_genre = books_complete_info
        self.ratings_df = ratings_df
        self.nn_model, self.tfidf_matrix, self.svd_model = self.get_nn_output(books_complete_info=books_complete_info)

    def get_nn_output(self, books_complete_info):

        print('Model 3')
        # Step 2: Train SVD for collaborative filtering
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['user_id', 'isbn', 'book_rating']], reader)
        trainset = data.build_full_trainset()

        svd_model = SVD()
        svd_model.fit(trainset)

        # Step 3: Prepare content-based filtering data
        books_complete_info['content_features'] = (
                books_complete_info['book_author'] + " " +
                books_complete_info['publisher'] + " " +
                books_complete_info['year_of_publication'].astype(str) + " " +
                books_complete_info['category']
        )

        # Apply TF-IDF and fit Nearest Neighbors
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(books_complete_info['content_features'])

        nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        nn_model.fit(tfidf_matrix)
        print('Model 3 return')
        return nn_model, tfidf_matrix, svd_model

    # Step 4: Hybrid Recommendation Function
    def hybrid_recommend_books(self, user_id, n_recommendations=5, svd_weight=0.7, content_weight=0.2,
                               above_median_weight=0.1):
        print('Model 2 reco')
        # Step 1: Get books the user has not interacted with
        all_books = self.books_missing_genre['isbn'].unique()
        rated_books = self.ratings_df[self.ratings_df['user_id'] == user_id]['isbn'].tolist()
        not_interacted_books = [isbn for isbn in all_books if isbn not in rated_books]

        # Step 2: Precompute SVD predictions for all not-interacted books
        svd_predictions = {
            isbn: self.svd_model.predict(user_id, isbn).est for isbn in not_interacted_books
        }

        # Step 3: Precompute content-based similarity for all not-interacted books
        user_books_indices = self.books_missing_genre.index[
            self.books_missing_genre['isbn'].isin(rated_books)
        ].tolist()
        not_interacted_books_indices = self.books_missing_genre.index[
            self.books_missing_genre['isbn'].isin(not_interacted_books)
        ].tolist()

        # Calculate pairwise similarities between the user's rated books and all not-interacted books
        if user_books_indices:
            distances, indices = self.nn_model.kneighbors(
                self.tfidf_matrix[not_interacted_books_indices], n_neighbors=len(user_books_indices)
            )
            similarity_scores = 1 - distances.mean(axis=1)
        else:
            similarity_scores = np.zeros(len(not_interacted_books))  # Default to 0 similarity if no rated books

        # Step 4: Precompute above-median score
        similar_books = self.ratings_df[self.ratings_df['isbn'].isin(rated_books)]
        above_median_score = (
            similar_books['above_median'].mean() if not similar_books.empty else 0
        )

        # Step 5: Combine all scores to calculate hybrid scores
        hybrid_scores = [
            (
                isbn,
                (svd_weight * svd_predictions[isbn]) +
                (content_weight * similarity_scores[i]) +
                (above_median_weight * above_median_score)
            )
            for i, isbn in enumerate(not_interacted_books)
        ]

        # Step 6: Sort by hybrid scores and return top recommendations
        recommendations = pd.DataFrame(hybrid_scores, columns=['isbn', 'hybrid_score'])
        recommendations = recommendations.sort_values(by='hybrid_score', ascending=False)

        print('AA')
        return recommendations.head(n_recommendations)
