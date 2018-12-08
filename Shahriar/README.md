# Extracting hd5 files into a single file and transforming them to single Numpy array


## Aggregating

I transformed all these files to python3. First, to add all hd5 to a single file, modify the maindir variable inside the main part of the "create_aggregate_file.py" to the data folders of hd5 and specify the output file name. After about 45 mins, you get the single aggregate hd5.

## Extracting data to a Numpy array

Now, read in the output file generated in the last section in "hdf5_getters_mod.py" file. Then, you'll get a numpy array in "result2" array. In "result_nonz" array, I removed any data that didn't specify the song year. 

The features in each column of data is as follows: 
1 get_analysis_sample_rate \n
2 get_artist_7digitalid
3 get_artist_familiarity
4 get_artist_hotttnesss
5 get_artist_latitude
6 get_artist_longitude
7 get_artist_playmeid
8 get_artist_terms_freq
9 get_artist_terms_weight
10 get_bars_confidence
11 get_bars_start
12 get_beats_confidence
13 get_beats_start
14 get_danceability
15 get_duration
16 get_end_of_fade_in
17 get_energy
18 get_key
19 get_loudness
20 get_mode
21 get_mode_confidence
22 get_release
23 get_release_7digitalid
24 get_sections_confidence
25 get_sections_start
26 get_segments_confidence
27 get_segments_loudness_max
28 get_segments_loudness_max_time
29 get_segments_loudness_start
30 get_segments_pitches
31 get_segments_start
32 get_segments_timbre
33 get_song_hotttnesss
34 get_song_id
35 get_start_of_fade_out
36 get_tatums_confidence
37 get_tatums_start
38 get_tempo
39 get_time_signature
40 get_time_signature_confidence
41 get_track_7digitalid
42 get_year
