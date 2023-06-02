import asyncio
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

print()
print(
"1. Sync\n" +
"2. Async"
)


async def async_prediction():
    await asyncio.create_task(hf.async_prediction(pattern))
while True:
    option = int(input("Option: "))
    if option == 1:
        # hf.print_pattern(pattern)
        hf.sync_prediction(pattern)
        break
    elif option == 2:
        # hf.print_pattern(pattern)
        asyncio.run(async_prediction())
        break
    else:
        print("Wrong option")






