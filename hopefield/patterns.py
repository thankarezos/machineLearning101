import numpy as np

patterns = [None] * 10


patterns[5] = np.array([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]])

patterns[6] = np.array([
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]])

patterns[8] = np.array([
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]])


patterns[9] = np.array([
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1,1, 1, 1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, 1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]])

patternsMod = [None] * 10

patternsMod[5] = np.array([[-1, -1, -1, -1, -1, -1, 1],
                      [-1, -1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, 1, 1, -1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, 1, -1]])

patternsMod[6] = np.array([
                      [-1, -1, -1, -1, -1, -1, 1],
                      [-1, -1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, -1, -1, -1, -1, -1],
                      [-1, 1, 1, -1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, -1, 1, 1, 1, -1, -1]])

patternsMod[8] = np.array([
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, 1, 1, 1, -1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, -1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, 1, 1, -1, -1]])


patternsMod[9] = np.array([
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, 1, 1, 1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, -1, -1, -1, 1, -1],
                      [-1, 1, 1, -1, 1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, -1, -1, -1, 1, -1],
                      [-1, -1, 1, -1, 1, 1, -1]])