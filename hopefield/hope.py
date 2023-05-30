from hopfieldnetwork import HopfieldNetwork
import patterns as pt

hopfield_network1 = HopfieldNetwork(N=4)

patterns = [pt.pattern_5.flatten().tolist(), pt.pattern_6.flatten().tolist(), pt.pattern_8.flatten().tolist(), pt.pattern_9.flatten().tolist()]
hopfield_network1.train_pattern(patterns)