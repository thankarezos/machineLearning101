import asyncio
import numpy as np
import patterns as pt

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
        num_neurons = np.prod(pattern.shape)
        pattern = pattern.reshape((num_neurons, 1))

        for _ in range(max_iterations):
            # output = np.dot(self.weights, pattern)
            output = np.dot(self.weights[:num_neurons, :num_neurons], pattern) 
            output[output >= 0] = 1
            output[output < 0] = -1

            if np.array_equal(output, pattern):
                return output.flatten().tolist()
            pattern = output
        
    async def update_neuron_async(self, neuron, pattern):
        num_neurons = np.prod(pattern.shape)
        weighted_sum = np.dot(self.weights[neuron, :num_neurons], pattern)
        output = np.where(weighted_sum >= 0, 1, -1)
        return np.atleast_1d(output)


    async def predict_async(self, pattern, max_iterations=100):
        pattern = np.array(pattern)
        num_neurons = np.prod(pattern.shape)
        pattern = pattern.reshape((num_neurons , 1))

        for _ in range(max_iterations):
            tasks = []
            for neuron in range(num_neurons):
                task = asyncio.create_task(self.update_neuron_async(neuron, pattern))
                tasks.append(task)
            outputs = await asyncio.gather(*tasks)

            output = np.concatenate(outputs, axis=0)
            output = np.where(output >= 0, 1, -1)

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

def print_pattern2(getPatterns):
    patterns = []
    for i in range(len(getPatterns)):
        if getPatterns[i] is not None:
            pattern = getPatterns[i].tolist()
            patterns.append(pattern)

    max_height = max(len(pattern) for pattern in patterns)

    for i in range(max_height):
        for pattern in patterns:
            if i < len(pattern):
                line = ""
                for element in pattern[i]:
                    if element == 1:
                        line += "  "
                    else:
                        line += "██"
                print(line, end='')
        print()

    return None

def print_pattern3(pattern1, pattern2):
    max_rows = max(len(pattern1), len(pattern2))

    for i in range(max_rows):
        line = ""

        if i < len(pattern1):
            row1 = pattern1[i]
            for element in row1:
                if element == 1:
                    line += "  "
                else:
                    line += "██"
            line += "  "  # Add spacing between patterns

        if i < len(pattern2):
            row2 = pattern2[i]
            for element in row2:
                if element == 1:
                    line += "  "
                else:
                    line += "██"

        print(line)

    return None



patterns = []

for i in range(len(pt.patterns)):
    if pt.patterns[i] is not None:
        pattern = pt.patterns[i].tolist()
        patterns.append(pattern)

network = HopfieldNetwork(num_neurons=77)
network.train(patterns)


# print_pattern(pattern)


def sync_prediction(pattern):
    result_sync = network.predict(pattern, max_iterations=200)  # Adjust pattern size to 10x7
    if result_sync:
        # pattern_size = int(np.sqrt(len(result_sync)))
        # print_pattern(np.array(result_sync).reshape((10, 7)))
        print_pattern3(pattern, np.array(result_sync).reshape((10, 7)))
    else:
        print("Synchronous pattern retrieval failed.")

# Asynchronous prediction

async def async_prediction(pattern):
    result_async = await network.predict_async(pattern, max_iterations=200)
    if result_async:
        print_pattern3(pattern, np.array(result_async).reshape((10, 7)))
    else:
        print("Asynchronous pattern retrieval failed.")