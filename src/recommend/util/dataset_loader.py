import os
from typing import Tuple

import pandas as pd


class DatasetLoader:
    def __init__(self, data_dir: str, user_nums: int | None = None):
        self.data_dir = data_dir
        self.user_nums = user_nums

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the all datasets

        Parameters
        ----------
            None

        Returns
        -------
            movies: pd.DataFrame
                movies(index, movie_id, title, genres, tags)
            ratings: pd.DataFrame
                ratings(index, user_id, movie_id, rating, timestamp)
        """
        movies = self.__load_movies()
        ratings = self.__load_ratings()
        tags = self.__load_tags()

        movies, ratings = self.__preprocess(movies, ratings, tags)

        print(movies)
        print(ratings)

        return movies, ratings

    def __preprocess(
        self, movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the datasets

        Parameters
        ----------
            movies: pd.DataFrame
                movies(index, movie_id, title, genres)
            ratings: pd.DataFrame
                ratings(index, user_id, movie_id, rating, timestamp)
            tags: pd.DataFrame
                tags(index, user_id, movie_id, tag, timestamp)

        Returns
        -------
            movies: pd.DataFrame
                movies(index, movie_id, title, genres, tags)
            ratings: pd.DataFrame
                ratings(index, user_id, movie_id, rating, timestamp)
        """

        # unify tags to lowercase
        tags["tag"] = tags.tag.str.lower()

        # associate tags with movies
        movie_with_tags = tags.groupby("movie_id").agg({"tag": list}).rename(columns={"tag": "tags"})
        movies = movies.merge(movie_with_tags, on="movie_id", how="left")

        if self.user_nums is None:
            return movies, ratings

        # limit the number of users
        valid_user_ids = sorted(ratings.user_id.unique())[: self.user_nums]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]

        return movies, ratings

    def __load_movies(self) -> pd.DataFrame:
        """
        Load the movies dataset

        Parameters
        ----------
            None

        Returns
        -------
            movies: pd.DataFrame
                movies(index, movie_id, title, genres)
        """
        movie_cols = ["movie_id", "title", "genres"]
        movies = pd.read_csv(
            os.path.join(self.data_dir, "movies.dat"), sep="::", names=movie_cols, encoding="latin-1", engine="python"
        )

        # convert type of genres from str to list
        movies["genres"] = movies.genres.apply(lambda x: x.split("|"))

        return movies

    def __load_ratings(self) -> pd.DataFrame:
        """
        Load the ratings dataset

        Parameters
        ----------
            None

        Returns
        -------
            ratings: pd.DataFrame
                ratings(index, user_id, movie_id, rating, timestamp)
        """
        rating_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            os.path.join(self.data_dir, "ratings.dat"), sep="::", names=rating_cols, encoding="latin-1", engine="python"
        )

        return ratings

    def __load_tags(self) -> pd.DataFrame:
        """
        Load the tags dataset

        Parameters
        ----------
            None

        Returns
        -------
            tags: pd.DataFrame
                tags(index, user_id, movie_id, tag, timestamp)
        """
        tag_cols = ["user_id", "movie_id", "tag", "timestamp"]
        tags = pd.read_csv(
            os.path.join(self.data_dir, "tags.dat"), sep="::", names=tag_cols, encoding="latin-1", engine="python"
        )

        return tags
