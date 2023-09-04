import requests
import json

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
    if "release_date" in result:
        context += f"was released in {result['release_date']}"
    if "genres" in result:
        genres_str = ""
        for genres in result["genres"]:
            genres_str += genres + ", "
        context += f"The genre of the movie is {genres_str}."
    if "stars" in result:
        context += f"has the actors/stars {', '.join(result['stars'])}, "
    if "directors" in result:
        context += f"directed by {', '.join(result['directors'])}, "
    if "producers" in result:
        context += f"produced by {', '.join(result['producers'])}. "
    if "plotLong" in result:
        context += f"The plot of the movie is {result['plotLong']}. "
    if "overview" in result:
        context += f"The plot of the movie is {result['overview']}. "
    if "location" in result:
        context += f"The movie was shot in the following locations: {', '.join(set(result['location']))}. "
    if "rating" in result:
        context += f"It has rating of {result['rating']}. "
    if "location" in result:
        context += f"The movie belongs to the genres {', '.join(result['genres'])}"

    return context
