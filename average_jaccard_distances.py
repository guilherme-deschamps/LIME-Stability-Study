from numpy import genfromtxt
import numpy as np

PATH = "./results/run 2/5000 samples/10 features/jaccard_distances.csv"
distances = genfromtxt(PATH, delimiter=',')
distances_without_zeros = distances[np.where(distances != 0)]

average = np.average(distances_without_zeros)
print(average)
