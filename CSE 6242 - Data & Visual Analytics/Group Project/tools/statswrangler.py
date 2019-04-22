import fileinput
import json

data = []
for line in fileinput.input():
    (count,pitch) = line.lstrip().rstrip().split(" ")
    data.append({"pitch" : pitch, "usage": int(count)})

print(json.dumps(data))
