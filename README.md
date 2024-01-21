# Various Recommendation Methods

## :movie_camera: Dataset
- [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/)
> MovieLens 10M movie ratings. Stable benchmark dataset. 10 million ratings and 100,000 tag applications applied to 10,000 movies by 72,000 users. Released 1/2009.

## :seedling: Recommendation Methods

- [x] Random
- [x] Association Rules
- [ ] User-based Collaborative Filtering
- [ ] Regression Model
- [ ] ...

## :computer: Required Tools

- [pyenv](https://github.com/pyenv/pyenv)
   > pyenv lets you easily switch between multiple versions of Python. It's simple, unobtrusive, and follows the UNIX tradition of single-purpose tools that do one thing well.

- [poetry](https://github.com/python-poetry/poetry)
   > Poetry helps you declare, manage and install > > dependencies of Python projects, ensuring you have the right stack everywhere.

## :gear: Initialization

1. Download MovieLens 10M Dataset in zip format from the following URL.
   - https://grouplens.org/datasets/movielens/10m/
2. Unzip the downloaded zip file at a location you desire.
3. Move the following files to `/dataset/movielens-10m/`.
   - `movies.dat`
   - `ratings.dat`
   - `tags.dat`
4. Install the required python packages.
   ```sh
   > poetry install
   ``` 
5. Set up git hooks
   ```sh
   > pre-commit install
   ```

## :warning: Note
- How to install python packages for each environment.
   ```sh
   > poetry install --with ${GROUP NAME}
   ```