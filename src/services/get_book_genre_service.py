import requests


def get_book_info(title, author, api_key=None):
    # Define the query with title and author
    query = f'intitle:{title}+inauthor:{author}'
    url = f'https://www.googleapis.com/books/v1/volumes?q={query}'

    # If you have an API key, add it to the URL
    if api_key:
        url += f'&key={api_key}'

    # Make the request
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data:
            # Get the first item in the search results
            book_info = data['items'][0]['volumeInfo']
            return {
                'title': book_info.get('title'),
                'authors': book_info.get('authors', []),
                'categories': book_info.get('categories', []),  # This contains genre/category information
                'description': book_info.get('description')
            }
        else:
            return "No results found"
    else:
        return f"Error: {response.status_code}"
