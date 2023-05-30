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
        pattern = pattern.reshape((self.num_neurons - 7, 1))

        for _ in range(max_iterations):
            # output = np.dot(self.weights, pattern)
            output = np.dot(self.weights[:70, :70], pattern) 
            output[output >= 0] = 1
            output[output < 0] = -1

            if np.array_equal(output, pattern):
                return output.flatten().tolist()
            pattern = output
        
    async def predict_async(self, pattern, max_iterations=100):
        pattern = np.array(pattern)
        pattern = pattern.reshape((self.num_neurons - 7, 1))

        for _ in range(max_iterations):
            output = np.dot(self.weights[:70, :70], pattern) 
            output[output >= 0] = 1
            output[output < 0] = -1

            if np.array_equal(output, pattern):
                return output.flatten().tolist()
            pattern = output
            await asyncio.sleep(0)  # Allow other tasks to run

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


patterns = []

for i in range(len(pt.patterns)):
    if pt.patterns[i] is not None:
        pattern = pt.patterns[i].tolist()
        patterns.append(pattern)

network = HopfieldNetwork(num_neurons=77)
network.train(patterns)

pattern = pt.patternsMod[9]
print_pattern(pattern)


result_sync = network.predict(pattern, max_iterations=200)  # Adjust pattern size to 10x7
if result_sync:
    # pattern_size = int(np.sqrt(len(result_sync)))
    print_pattern(np.array(result_sync).reshape((10, 7)))
else:
    print("Synchronous pattern retrieval failed.")

# Asynchronous prediction
async def async_prediction():
    result_async = await network.predict_async(pattern, max_iterations=200)
    if result_async:
        print_pattern(np.array(result_async).reshape((10, 7)))
    else:
        print("Asynchronous pattern retrieval failed.")

loop = asyncio.get_event_loop()
loop.run_until_complete(async_prediction())
loop.close()