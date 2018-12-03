# CSE 5523 Final Project

# Installation
1. Get the data: https://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset
2. Install the dependencies of the python analysis scripts: http://www.pytables.org/usersguide/installation.html
3. Symlink the data directory in the million song dataset into the 5523 project directory at `data`

# Details
* Spotify client ID (if we want to use their API for anything):
    * Client ID: 584581d4820e484c90b1e2cdf9f13cee
    * Client Secret: c73487557a454e9b9bdbcc6f57786256
* HDF5 helper scripts: https://github.com/tbertinmahieux/MSongsDB
* Echo nest (used for original segmentation) and 7digital (used to get 30 second snippets as audio files0 no longer support development api's, but looks like same/similar features can be extracted through spotify API (they acquired echo nest): https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/

# Classifiers:
* Boosting:
    * AdaBoost via scikit: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    * Which uses these people's stuff (fancy decision stumps) as decision stumps I think: http://www.multimedia-computing.de/fertilized/
