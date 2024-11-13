
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.handler.get_all_data_handler import GetAllData
from src.services.get_genre_of_books_service import get_genre_of_books

# Get All data
source_data_all = GetAllData()
books_df = source_data_all.load_books()
ratings_df = source_data_all.load_ratings()
users_df = source_data_all.load_users()

# Data pre-processing
# Convert "NULL" strings to NaN in the age column
users_df['age'] = users_df['age'].replace("NULL", np.nan)
users_df['age'] = users_df['age'].astype(float)
users_df['age'] = users_df['age'].replace(0, users_df['age'].median())

# Replace invalid years (e.g., 0) with NaN, then fill or drop as appropriate
books_df['year_of_publication'] = books_df['year_of_publication'].fillna(books_df['year_of_publication'].median())
books_df['publisher'] = books_df['publisher'].fillna("Unknown Publisher")
books_df['book_author'] = books_df['book_author'].fillna("Unknown")

books_model_df = books_df[['isbn', 'book_title', 'book_author', 'year_of_publication', 'publisher']]

ratings_df['book_rating'] = ratings_df['book_rating'].replace(0, np.nan)
valid_ratings_df = ratings_df.loc[ratings_df['book_rating'] > 0]
# Calculate the median rating for each book (grouped by ISBN)
book_median_ratings = valid_ratings_df.groupby('isbn')['book_rating'].median().rename('median_rating').reset_index()
# Merge median rating with the ratings' DataFrame
ratings_df = ratings_df.merge(valid_ratings_df, on='isbn')
# Create an indicator column for above/below median rating
ratings_df['above_median'] = (valid_ratings_df['book_rating'] > valid_ratings_df['median_rating']).astype(int)
# Count the number of reviews for each book
review_counts = valid_ratings_df.groupby('isbn').size().rename('review_count').reset_index()
# Filter the DataFrame to keep only books with at least 3 reviews
review_counts_three = review_counts[review_counts['review_count'] >= 3]


# books_filter_genre_df = get_genre_of_books(books_model_df=books_model_df,
#                                            review_counts_list=list(review_counts_three['isbn']))

# As we want to be sure to not only rely on ratings, we will select a hybrid approach where we

# Load data into Surprise format
reader = Reader(rating_scale=(1, 10))  # Adjust if needed
data = Dataset.load_from_df(valid_ratings_df[['user_id', 'isbn', 'book_rating']], reader)

# Split into training and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=10)

# Initialize and train the SVD model
svd_model = SVD()
svd_model.fit(trainset)

# Combine content features into a single string for each book
books_df['content_features'] = books_df['author'] + " " + books_df['publisher'] + " " + books_df['year_of_publication'].astype(str)

# Apply TF-IDF transformation
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books_df['content_features'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get content similarity score
def get_content_similarity(isbn1, isbn2):
    idx1 = books_df.index[books_df['isbn'] == isbn1][0]
    idx2 = books_df.index[books_df['isbn'] == isbn2][0]
    return cosine_sim[idx1, idx2]

# So, the usecase is, that we have book selling e-store that want personalise marketing. On our data, we should devide books as good and bad, based on the rating and than find similar features of the books. As Author,
breakpoint()