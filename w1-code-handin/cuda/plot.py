import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data.csv", sep=";")
print(df)
valid = df[df["validity"] == 1]
invalid = df[df["validity"] == 0]

print("--- Valid ----")
print(valid)

print("--- Invalid ----")
print(invalid)

#x = valid["array_size"]
#y = valid["speedup"]
#
#plt.plot(x, y)
#plt.savefig("speedup.png")

