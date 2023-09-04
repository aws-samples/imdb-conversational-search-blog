"""
This script helps create an opensearch index. Use the command:

python index_creation.py --host [opensearch host id] --region [region name] --imdb_file [s3 path to parquet file] --index_name [name of index]
"""
import numpy as np
import json
import pandas as pd
import argparse

import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection


def initialize_ops(host, region, port=443, service="es"):
    """
    Create an opensearch instance with the corresponding arguments.
    Args:
        host(str): opensearch host id
        region(str): region name
    Return:
        OpenSearch: initialized opensearch instance
    """
    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token,
    )

    ops = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=100000,
    )

    return ops


def create_index(index, ops):
    """
    This function will create an index using the knn settings
    Args:
        index(str): index name
    Returns:
        None: Creates an elasticsearch index
    """
    if not ops.indices.exists(index=index):
        index_settings = {
            "settings": {
                "index.knn": True,
                "index.knn.space_type": "cosinesimil",
                "analysis": {
                    "analyzer": {
                        "default": {"type": "standard", "stopwords": "_english_"}
                    }
                },
            },
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": 512,  # replace with your embedding size
                    }
                }
            },
        }

        ops.indices.create(index=index, body=json.dumps(index_settings))
        print("Created the elasticsearch index successufly ")
    else:
        print("elasticsearch index already exists")


def post_request_emb(index_name, movies, ops):
    """
    Creates the json string for bulk load for KNN index
    Args:
        index_name(str): name of the opensearch index
        movies(List(Dict)): full movie data
        ops(OpenSearch): Initialized opensearch instance
    Returns:
        Dict: response from the bulk load
    """
    data = ""
    for movie in movies:
        data += (
            '{ "index": { "_index": "'
            + index_name
            + '", "_id": "'
            + movie["id"]
            + '" } }\n'
        )
        data += '{ "embeddings": ' + str(list(movie["embeddings"])) + "}\n"
        # data += '{ "year": '+ movie['year'] + ', "poster": "' + movie['poster'] + '","embeddings": ' + str(movie['embeddings']) + ', "title": "' + movie['title']+'"}\n'
    response = ops.bulk(data)
    return response


def ingest_data_into_ops_knn(
    combined_data, files, ops, ops_index="ooc_knn", post_method=post_request_emb
):
    """
    Ingests data into an Opensearch KNN index
    Args:
        combined_data(list): includes ids of movies
        files(list): includes embedding and title ids of movies
        ops(OpenSearch): initialized opensearch index
        ops_index(str): name of opensearch index
        post_method(<function>): Function to bulk load movies into OpenSearch index
    Returns:
        Dict: response from the post method output
    """
    movies, i = [], 1
    for ids, (embedding, tt_ids) in enumerate(zip(combined_data, files)):
        # movie={'id': tt_ids, 'embeddings': embedding, 'title': name,'year':str(year),'poster':poster}
        movie = {"id": tt_ids[21:-8], "embeddings": embedding}
        movies.append(movie)
        if i % 10000 == 0:
            response = post_method(ops_index, movies, ops)
            print(f"Processing line {i}")
            movies = []
        i += 1
    response = post_method(ops_index, movies, ops)

    return response


def post_request(data, ops):
    """
    Bulk request for ingesting data to opensearch
    Args:
        ops(OpenSearch): initialized opensearch index
        data(List): bulked data to upload into opensearch
    Returns:
        Dict: response from the post method output
    """
    response = ops.bulk(data)
    return response


def ingest_data_into_ops_text(df, ops, ops_index, post_method):
    """
    Ingest data into an Opensearch text index
    Args:
        df(Pandas Dataframe): dataframe of full preprocessed movies dataset
        ops(OpenSearch): initialized opensearch index
        ops_index(str): name of opensearch index
        post_method(<function>): Function to bulk load movies into OpenSearch index
    Returns:
        Dict: response from the post method output
    """
    data = ""
    for i, (
        tt_ids,
        originalTitle,
        genres,
        year,
        actors,
        directors,
        producers,
        keyword,
        location,
        plotLong,
        rating,
        poster_url,
    ) in enumerate(df.values):
        data += (
            '{ "index": { "_index": "' + ops_index + '", "_id": "' + tt_ids + '" } }\n'
        )
        data += "{ "
        originalTitle = (
            originalTitle.replace("'", "").replace('"', "").replace("\\", "")
        )
        data += f'"title": "{originalTitle}"'
        if year != "null":
            data += f', "year": {int(year)}'
        if plotLong != "null":
            plotLong = (
                json.dumps(plotLong)
                .replace('"', "")
                .replace("'", "")
                .replace("\\n", "")
                .replace("\\", "")
            )
            data += f', "plotLong": "{plotLong}"'
        if rating != "null":
            data += f', "rating": {rating}'
        if isinstance(genres, np.ndarray):
            data += f', "genres": {json.dumps(list(genres))}'
        if actors.size > 0:
            actors = json.dumps([a.replace("'", "").replace('"', "") for a in actors])
            data += f', "stars": {actors}'
        if directors.size > 0:
            directors = json.dumps(
                [a.replace("'", "").replace('"', "") for a in directors]
            )
            data += f', "directors": {directors}'
        if producers.size > 0:
            producers = json.dumps(
                [a.replace("'", "").replace('"', "") for a in producers]
            )
            data += f', "producers": {producers}'
        if keyword.size > 0:
            keyword = json.dumps([k.replace("'", "") for k in keyword])
            data += f', "keywords": {keyword}'
        if location.size > 0:
            location = json.dumps([k.replace("'", "") for k in location])
            data += f', "location": {location}'
        if poster_url != "":
            data += f', "poster_url": "{poster_url}"'
        #         data += ', "embeddings": ' + str(mapped_emb[tt_ids])

        data += " }\n"
        if i % 10000 == 0:
            # data = data.replace("\'",'\"')
            # print(data)
            response = post_method(data, ops)
            if response["errors"]:
                print(
                    [
                        x["index"]["_id"]
                        for x in response["items"]
                        if x["index"]["status"] != 200
                    ]
                )
                print(response, data, plotLong.replace('"', ""))
                break
            print(f"Processing line {i}")
            data = ""
        i += 1
    response = post_method(data, ops)
    return response


def load_data_into_os(args):
    """
    Load all the data into opensearch clusters including the KNN and Text index
    Args:
        args(ArgumentParser): user inputted or defaulted arguments
    Returns:
        dict: response from the ops_text input function
        boolean: response from the ops_knn input function
    """
    ops = initialize_ops(args.host, args.region)
    df_meta = pd.read_parquet(args.imdb_file)

    df_meta["year"] = df_meta["year"].apply(lambda x: int(x) if ~np.isnan(x) else x)
    df_meta = df_meta.fillna("null")
    print("creating exact match index.......")
    response = ingest_data_into_ops_text(
        df_meta, ops, ops_index=args.index_name, post_method=post_request
    )
    print("creating knn/semantic match index.......")
    knn_response = ingest_data_into_ops_knn(ops, args)
    return response, knn_response


def ingest_data_into_ops_knn(ops, args):
    """
    Load all the data into opensearch clusters using a KNN index
    Args:
        ops(OpenSearch): initialized opensearch index
        args(ArgumentParser): user inputted or defaulted arguments
    Returns:
        boolean: whether KNN index was created
    """
    movies = pd.read_parquet(args.embedding_parquet)

    knn_index = {
        "settings": {
            "index.knn": True,
            "index.knn.space_type": "cosinesimil",
            "analysis": {
                "analyzer": {"default": {"type": "standard", "stopwords": "_english_"}}
            },
        },
        "mappings": {
            "properties": {
                "emb": {"type": "knn_vector", "dimension": 768, "store": True},
                "directors": {"type": "text", "store": True},
                "producers": {"type": "text", "store": True},
                "stars": {"type": "text", "store": True},
                "rating": {"type": "text", "store": True},
                "location": {"type": "text", "store": True},
                "genres": {"type": "text", "store": True},
                "year": {"type": "text", "store": True},
                "plotLong": {"type": "text", "store": True},
                "title": {"type": "text", "store": True},
                "poster_url": {"type": "text", "store": True},
                "titleId": {"type": "text", "store": True},
                "keyword": {"type": "text", "store": True},
            }
        },
    }
    try:
        ops.indices.create(index=args.knn_index_name, body=knn_index, ignore=400)
        for row in movies.itertuples():
            ops.index(
                index=args.knn_index_name,
                body={
                    "emb": row.plot_keyword_emb,
                    "titleId": row.titleId,
                    "title": row.title,
                    "poster_url": row.poster_url,
                    "directors": row.directors,
                    "producers": row.producers,
                    "stars": row.stars,
                    "rating": row.rating,
                    "location": row.location,
                    "genres": row.genres,
                    "year": row.year,
                    "plotLong": row.plotLong,
                    "keyword": row.all_keywords,
                },
            )
        print("knn index created")
        res = ops.search(index=args.knn_index_name, body={"query": {"match_all": {}}})
        print("Records found: %d." % res["hits"]["total"]["value"])
        return True
    except Exception as err:
        print(err)
        return False


def parse_args():
    """
    Load all args from the call to the script. This includes the host id, region,
        imdb file, and index name.
    """
    parser = argparse.ArgumentParser(description="Create opensearch index")
    parser.add_argument(
        "--host",
        type=str,
        default="vpc-llm-imdb-search-wdv7w5ydpuuzhekveyxmnhsnw4.us-east-1.es.amazonaws.com",
        help="opensearch host id",
    )
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="opensearch region id"
    )
    parser.add_argument(
        "--imdb_file",
        type=str,
        default="s3://mlsl-imdb-data/imdb_ml_10k_posters.parquet",
        help="imdb metadata file",
    )
    parser.add_argument(
        "--embedding_parquet",
        type=str,
        default="s3://mlsl-imdb-data/plot_keyword_embeddings.parquet",
        help="imdb metadata file",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="imdb_small_posters",
        help="imdb metadata file",
    )
    parser.add_argument(
        "--knn_index_name", type=str, default="imdb_plot_knn", help="imdb metadata file"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    load_data_into_os(args)
