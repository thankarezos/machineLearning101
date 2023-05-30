import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        
    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            pattern = pattern.reshape((self.num_neurons, 1))
            self.weights += np.dot(pattern, pattern.T)
            np.fill_diagonal(self.weights, 0)

        # Normalize weights
        self.weights /= len(patterns)

    def predict(self, pattern, max_iterations=100):
        pattern = np.array(pattern)
        pattern = pattern.reshape((self.num_neurons, 1))

        for _ in range(max_iterations):
            output = np.dot(self.weights, pattern)
            output[output >= 0] = 1
            output[output < 0] = -1

            if np.array_equal(output, pattern):
                return output.flatten().tolist()
            pattern = output

def print_pattern(pattern):
    for row in pattern:
        line = ""
        for element in row:
            if element == 1:
                line += "  "
            else:
                line += "██"
        print(line)

    return None

# Example usage
pattern_5 = np.array([[-1, -1, -1, -1, -1, -1, -1],
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

pattern_6 = np.array([
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

pattern_8 = np.array([
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


pattern_9 = np.array([
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


patterns = [pattern_5.flatten().tolist(), pattern_6.flatten().tolist(), pattern_8.flatten().tolist(), pattern_9.flatten().tolist()]

network = HopfieldNetwork(num_neurons=len(patterns[0]))
network.train(patterns)

retrieved_pattern = network.predict(pattern_9)
if retrieved_pattern:
    pattern_size = int(np.sqrt(len(retrieved_pattern)))
    print_pattern(np.array(retrieved_pattern).reshape((11, 7)))
    # print(np.array(retrieved_pattern).reshape((11, 7)))
else:
    print("Pattern retrieval failed.")
