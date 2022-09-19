import subprocess

array_sizes = [750000, 800000, 900000, 100000000]
block_sizes = [256, 512]
epsilons = [0.0001, 0.001, 0.01, 0.1] 


for size in array_sizes:
    for block in block_sizes:
        for epsilon in epsilons:
            command = ["./wa1-task3", size, block, epsilon]
            subprocess.run(command)
             





