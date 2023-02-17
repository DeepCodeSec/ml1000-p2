#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import logging
import numpy as np
import pandas as pd
from pycaret.classification import *
#
logger = logging.getLogger(__name__)
#


class SomeDataset(object):
    """ Class wrapper around the selected dataset. """

    def __init__(self, _file:str, _target_col:str, _training_size=0.8, _drop_cols=[]) -> None:
        self._datafile = _file
        self._df = pd.read_csv(_file, sep=';')
        raise Exception("Not implemented.")

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