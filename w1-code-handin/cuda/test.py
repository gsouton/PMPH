import subprocess
import pandas as pd

array_sizes = [500000, 600000, 750000, 800000, 900000, 1000000, 10000000]
block_sizes = [256, 512, 1024]
epsilons = [0.00001, 0.0001, 0.001, 0.01, 0.1] 



subprocess.run(["make", "clean"])
subprocess.run(["make"])

data = {"array_size": [], "block_size": [], "epsilon": [], "cpu_time": [], "gpu_time": [], "speedup": [], "validity": []}

for size in array_sizes:
    print(f"- Benchmarking with array size = {size}")
    for block in block_sizes:
        print(f"\t- block size = {block}")
        for epsilon in epsilons:
            print(f"\t\t- epsilon = {epsilon}")
            command = ["./wa1-task3", str(size), str(block), str(epsilon)]
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            data["array_size"].append(size)
            data["block_size"].append(block)
            data["epsilon"].append(epsilon)

            line = proc.stdout.readline()
            data["cpu_time"].append(float(line))

            line = proc.stdout.readline()
            data["gpu_time"].append(float(line))

            line = proc.stdout.readline()
            data["speedup"].append(int(line))

            line = proc.stdout.readline()
            data["validity"].append(int(line))

print(data)
df = pd.DataFrame(data=data)
df.to_csv("data.csv", sep=";")
