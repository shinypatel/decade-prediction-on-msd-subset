# Extracting hd5 files into a single file and transforming them to single Numpy array


## Aggregating

I transformed all these files to python3. First, to add all hd5 to a single file, modify the maindir variable inside the main part of the "create_aggregate_file.py" to the data folders of hd5 and specify the output file name. After about 45 mins, you get the single aggregate hd5.

## Extracting data to a Numpy array

Now, read in the output file generated in the last section in "hdf5_getters_mod.py" file. Then, you'll get a numpy array in "result2" array. In "result_nonz" array, I removed any data that didn't specify the song year. 

