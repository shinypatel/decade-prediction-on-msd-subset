#!/bin/python3

import os
import fnmatch
import sqlite3
import hdf5_getters as hdf


class Dataset():
    def __init__(self):
        self.conn = sqlite3.connect('db/subset_track_metadata.db')
        self.c = _CONN.cursor()

    def get_training(self):
        pass



def get_hdf5(pattern):
    pattern += ".h5"
    directory = 'data/'
    f_list = []
    for d_name, sd_name, f_list in os.walk(directory):
        for file_name in f_list:
            if fnmatch.fnmatch(file_name, pattern):
                return hdf.open_h5_file_read(os.path.join(d_name, file_name))
