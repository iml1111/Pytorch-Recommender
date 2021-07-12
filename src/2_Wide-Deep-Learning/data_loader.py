import os
from itertools import product
import pandas as pd
from sklearn.model_selection import train_test_split


class KMRDDataLoader:
    """Load KMRD Dataset"""

    def __init__(self, data_path, valid_ratio=0.001):
        self.data_path = data_path
        self.valid_ratio = valid_ratio
        
        train_df, val_df = self._read_rates_df()
        movies_df = self._read_movies_df()

        #train_df = train_df[:1000] # check

        dummy_genres_df = movies_df['genres'].str.get_dummies(sep='/')
        train_genres_df = train_df['movie'].apply(lambda x: dummy_genres_df.loc[x])

        dummy_grade_df = pd.get_dummies(movies_df['grade'], prefix='grade')
        train_grade_df = train_df['movie'].apply(lambda x: dummy_grade_df.loc[x])

        train_df['year'] = train_df.apply(lambda x: movies_df.loc[x['movie']]['year'], axis=1)
        
        # 학습 데이터
        self.train_df = pd.concat([train_df, train_grade_df, train_genres_df], axis=1)
        # wide model에 들어갈 sparse하게 들어갈 칼럼
        self.wide_cols = list(dummy_genres_df.columns) + list(dummy_grade_df.columns)
        # wide model에 cross product로 들어갈 칼럼
        # cross product의 경우의 수가 너무 많으므로 일부분(3개)로만 구성
        self.cross_cols = self._make_cross_col(self.wide_cols[:3]) 
        # deep model에 embedding layer로 들어갈 칼럼
        self.embed_cols = list(set([(x[0], 16) for x in self.cross_cols]))
        # deep 모델에 linear layer로 들어갈 칼럼
        self.continuous_cols = ['year']
        # 정답 데이터 
        self.target = train_df['rate'].apply(lambda x: 1 if x > 9 else 0).values


    def _read_rates_df(self):
        return train_test_split(
            pd.read_csv(
                os.path.join(self.data_path, 'rates.csv')
            ),
            test_size=self.valid_ratio,
            random_state=1111, # random fix
            shuffle=True
        )

    def _read_movies_df(self):
        movies_df = pd.read_csv(
            os.path.join(self.data_path, 'movies.txt'), 
            sep='\t', encoding='utf-8'
        )
        movies_df = movies_df.set_index('movie')

        castings_df = pd.read_csv(
            os.path.join(self.data_path, 'castings.csv'), encoding='utf-8')
        countries_df = pd.read_csv(
            os.path.join(self.data_path, 'countries.csv'), encoding='utf-8')
        genres_df = pd.read_csv(
            os.path.join(self.data_path, 'genres.csv'), encoding='utf-8')

        # Get genre information
        genres = [
            (list(set(x['movie'].values))[0], '/'.join(x['genre'].values)) 
            for index, x in genres_df.groupby('movie')
        ]
        combined_genres_df = pd.DataFrame(data=genres, columns=['movie', 'genres'])
        combined_genres_df = combined_genres_df.set_index('movie')

        # Get castings information
        castings = [
            (list(set(x['movie'].values))[0], x['people'].values) 
            for index, x in castings_df.groupby('movie')
        ]
        combined_castings_df = pd.DataFrame(data=castings, columns=['movie','people'])
        combined_castings_df = combined_castings_df.set_index('movie')

        # Get countries for movie information
        countries = [
            (list(set(x['movie'].values))[0], ','.join(x['country'].values)) 
            for index, x in countries_df.groupby('movie')
        ]
        combined_countries_df = pd.DataFrame(data=countries, columns=['movie', 'country'])
        combined_countries_df = combined_countries_df.set_index('movie')

        return pd.concat(
            [
                movies_df, 
                combined_genres_df, 
                combined_castings_df, 
                combined_countries_df
            ], 
            axis=1
        )

    def _make_cross_col(self, wide_cols):
        unique_combinations = list(
            list(zip(wide_cols, element)) 
            for element in product(
                wide_cols, 
                repeat = len(wide_cols)
            )
        ) 

        cross_cols = [item for sublist in unique_combinations for item in sublist]
        cross_cols = [x for x in cross_cols if x[0] != x[1]]
        cross_cols = list(set(cross_cols))
        return cross_cols


