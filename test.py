import pstats

# Create a Stats object
stats = pstats.Stats('/home/skyr/Downloads/self_play_profiling.txt')

# Strip directories from file paths in the report, making it easier to read
stats.strip_dirs()

# Sort the data by cumulative time in the function
stats.sort_stats('cumulative')

# Print out the first 10 lines of the report
stats.print_stats(10)
