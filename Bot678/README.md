Run bot678.py and remember to maximize the plot window and set tight layout from the options in the plot window.
Since the data is gathered in a parallel fashion, please make sure to set the appropriate amount of jobs.
The number of jobs must be a factor of the number of alpha values data is being gathered for. The runs per alpha
and maximum turns allowed are both available to be set in the WorldState class. Make sure that BB\_SIZE
and BB\_SIZE\_4D have appropriate values. 40 and 60 worked well for us, but depending on the range of alpha values for which data is being gathered modification may be necessary.