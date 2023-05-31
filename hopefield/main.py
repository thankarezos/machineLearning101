import myHopefield as hf
import patterns as pt

# hf.print_pattern(pt.patterns[5])
print("Saved patterns:")
hf.print_pattern2(pt.patterns)


print("Chose pattern to recall")
hf.print_pattern2(pt.patternsMod)

while True:
    option = int(input("Give Number: "))
    if pt.patternsMod[option] is None:
        print("Pattern not found")
    else:
        break
pattern = pt.patternsMod[option]

hf.print_pattern(pattern)
hf.sync_prediction(pattern)