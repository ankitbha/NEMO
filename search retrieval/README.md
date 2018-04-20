Code for Search Retrieval System--------------

run the code as:
python  finalcode.py ./im_arousal.txt ./im_valence.txt ./static_annotations_averaged_songs_1_2000.csv ./static_annotations_averaged_songs_2000_2058.csv

on the terminal.
This gives the result for search part.
Once model is saved, 1st division (above the broken line in finalcode.py) can be commmented out to save time.

join_and_get.py joins the two csv files together to get the data for audio part. No need to run separately as is called by finalcode.

im_arousal and im_valence are files of the dataset as are the two csv files.

Important : Uses annoy library
Get the library as:
sudo pip install annoy


For further information about the library visit : https://github.com/spotify/annoy