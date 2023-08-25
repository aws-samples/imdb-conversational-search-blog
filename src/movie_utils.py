import requests
import json

def get_genre_ids(api_key):
    """
    Retrieves all possible genres and their corresponding ids.
    Args:
        api_key(str): tmdb api key to access tmdb information
    Returns:
        list(dict): contains a genre's name and its corresponding id.
    """
    # Construct the API request URL
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}&language=en-US"
    # Send the API request and retrieve the response as JSON
    response = requests.get(url)
    data = response.json()
    # Extract the list of genres
    genre_list = data.get("genres", [])
    genre_dict = {}
    for genre_entry in genre_list:
        genre_dict[genre_entry["id"]] = genre_entry["name"]
    return genre_dict

def decode_genre_ids(entry, genre_id_dict):
    """
    Decode genre ids to genre types
    Args:
        entry(dict): given movie information (includes genre ids)
        genre_id_dict(dict): mapping between genre id and genre type
    Returns:
        dict: include a genre key with the actual genre types
    """
    if "genre_ids" in entry:
        genres = []
        for genre_id in entry.get("genre_ids", []):
            genres.append(genre_id_dict.get(genre_id, genre_id))
        entry["genres"] = genres
    return entry

def describe_movie(result):
    """
    Process opensearch movie metadata into text description, used for chat context
    Args:
        result(dict): information about a particular movie
    Returns:
        str: text description of the movie from all aggregated information
    """
    context = ""
    if "title" in result:
        context += f"The name of the movie is {result['title']}, "
    if "year" in result:
        context += f"was shot in {result['year']}, "
    if "stars" in result:
        context += f"has the actors/stars {', '.join(result['stars'])}, "
    if "directors" in result:
        context += f"directed by {', '.join(result['directors'])}, "
    if "producers" in result:
        context += f"produced by {', '.join(result['producers'])}. "
    if "plotLong" in result:
        context += f"The plot of the movie is {result['plotLong']}. "
    if "location" in result:
        context += f"The movie was shot in the following locations: {', '.join(set(result['location']))}. "
    if "rating" in result:
        context += f"It has rating of {result['rating']}. "
    if "location" in result:
        context += f"The movie belongs to the genres {', '.join(result['genres'])}"

    return context

def get_trending_content(api_key):
    """
    Retrieves the movies and tv shows that are trending or popular today.
    Args:
        None
    Returns:
        list(dict): includes title and description of each trending movie or TV show.
    """
    # Construct the API request URL
    url = f"https://api.themoviedb.org/3/trending/all/day?api_key={api_key}"
    # Send the API request and retrieve the response as JSON
    response = requests.get(url)
    data = response.json()
    # Extract the list of movies from the response data
    movie_list = data.get("results", [])
    # Extract the title and description of each movie
    relevant_fields = [
        "original_language",
        "overview",
        "release_date",
        "title",
        "id",
        "genres",
    ]
    movies = []
    genre_id_dict = get_genre_ids(api_key)
    for entry in movie_list:
        entry = decode_genre_ids(entry, genre_id_dict)
        movies.append(
            {key: entry[key] for key in entry if key in relevant_fields}
        )
    movies = movies[:10]
    return json.dumps(movies)

def get_popular_movies(year):
    """
    Retrieves the popular movies of a given year along with their descriptions.
    Args:
        year (int): The year to search for popular movies.
    Returns:
        list(dict): includes the title and description of a popular movie from
            the given year.
    """
    # Construct the API request URL
    url = f"https://api.themoviedb.org/3/discover/movie?api_key={tmdb_api_key}&sort_by=popularity.desc&include_adult=false&include_video=false&primary_release_year={year}"
    # Send the API request and retrieve the response as JSON
    response = requests.get(url)
    data = response.json()
    # Extract the list of movies from the response data
    movie_list = data.get("results", [])
    # Extract the title and description of each movie
    relevant_fields = [
        "original_language",
        "overview",
        "release_date",
        "title",
        "id",
        "genres",
    ]
    movies = []
    for entry in movie_list:
        entry = decode_genre_ids(entry)
        movies.append(
            {key: entry[key] for key in entry if key in relevant_fields}
        )
    movies = movies[:10]
    return json.dumps(movies)
