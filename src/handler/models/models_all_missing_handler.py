
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class ModelAllMissingHandler:
    def __init__(self, books_missing_both):
        self.books_missing_both = books_missing_both
        self.nn_model, self.tfidf_matrix = self.get_nn_output(books_missing_both=books_missing_both)

    def get_nn_output(self, books_missing_both):
        # Cosine similarity overflowing memory -> select NearestNeighbors
        # Step 1: Combine metadata into a single feature
        books_missing_both['content_features'] = (
                books_missing_both['book_author'] + " " +
                books_missing_both['publisher'] + " " +
                books_missing_both['year_of_publication'].astype(str)
        )

        # Step 2: Apply TF-IDF to the content features
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(books_missing_both['content_features'])

        # Step 3: Fit Nearest Neighbors model
        nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        nn_model.fit(tfidf_matrix)

        return nn_model, tfidf_matrix


    def recommend_similar_books(self, book_isbn, n_recommendations=5):
        # Find the index of the given book
        idx = self.books_missing_both.index[self.books_missing_both['isbn'] == book_isbn][0]

        # Get the indices of the nearest neighbors
        distances, indices = self.nn_model.kneighbors(self.tfidf_matrix[idx], n_neighbors=n_recommendations + 1)

        # Exclude the input book itself (index 0 in the result)
        similar_books = self.books_missing_both.iloc[indices.flatten()[1:]]

        return similar_books

