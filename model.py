#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import logging
import numpy as np
import pandas as pd
from pycaret.clustering import *
#
logger = logging.getLogger(__name__)
#


class WhiteWineQualityDataset(object):
    """ Class wrapper around the selected dataset. """

    def __init__(self, _file:str, _training_size=0.8, _drop_cols=[]) -> None:
        self._datafile = _file
        self._df = pd.read_csv(_file, sep=';')

        # Remove/clean data items
        self._df = self._df.drop(columns=["Quality"])
        # Capping of outliers
        cols = list(self._df.columns)
        tmp = self._df #creating a temporary to avoid accidentally overwriting the original (let's us compare and verify capping)
        data_clean = self._df
        for col in cols[0:-1]:
            upper_limit = self._df[col].mean() + 3*self._df[col].std() #~95th percentile
            lower_limit = self._df[col].mean() - 3*self._df[col].std() #~5th percentile
            logger.info(f"[{col}] 5th and 95th percentiles identified: {upper_limit} - {lower_limit}")

            data_clean[col] = np.where(tmp[col]> upper_limit, upper_limit, #if above 95th, set to upper
                                    np.where(tmp[col]< lower_limit, lower_limit, #if below 5th, set to lower
                                    tmp[col]))

        # Create the classifier
        # Pass the complete dataset as data and the featured to be predicted as target
        self._clf = setup(
            data=data_clean, #make sure to use cleaned data (outliers capped)
            transformation=True, #applies the power transform to make data more Gaussian-like
            normalize=True, #transforms the numeric features by scaling them to a given range (default is z-score)
            remove_multicollinearity=True, #features with the inter-correlations higher than the defined threshold are removed
            multicollinearity_threshold = 0.7 #by default was 0.9
        )
        #ml_model = create_model(model='kmeans', num_clusters=2)
        self._best_model = create_model(model='kmeans', num_clusters=2)
        logger.info(self._best_model)

    @property
    def filename(self) -> str:
        return self._datafile

    @property
    def classifier(self):
        return self._clf

    @property
    def dataframe(self):
        return self._df

    @property
    def nb_rows(self) -> int:
        return self.dataframe.shape[0]

    @property
    def nb_cols(self) -> int:
        return self.dataframe.shape[1]

    @property
    def best_model(self):
        return self._best_model

    def save_best_model_to(self, _filename:str) -> None:
        # Save the best model to a file
        save_model(self.best_model, _filename)