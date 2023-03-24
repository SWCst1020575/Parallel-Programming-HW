import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
start = time.time()
end = time.time()
x1 = []
x2 = []
y = []


def test(pos1, pos2, size):
    start = time.time()
    os.system(
        f"srun -n4 -c4 ./hw2 4 {pos1} {pos2} {size} a.png")
    end = time.time()
    x1.append(end-start)
    start = time.time()
    os.system(
        f"srun -n4 -c4 ./hw2_default 4 {pos1} {pos2} {size} a.png")
    end = time.time()
    x2.append(end-start)


testcase = [["-0.522 2.874 1.340", "0 0 0", "64 64"],
            ["4.152 2.398 -2.601", "0 0 0", "128 128"],
            ["1.885 -1.570 3.213", "0 0 0", "512 512"],
            ["-0.027 -0.097 3.044", "0 0 0", "512 512"],
            ["3.726 0.511 -0.096", "0 0 0", "512 512"],
            ["0.7725 -0.385 1.3065", "0.782 -0.178 0.312", "1024 1024"],
            ["1.1187 -1.234 -0.285", "-0.282 -0.312 -0.378", "1024 1024"],
            ["1.1645 2.0475 1.7305", "-0.8492 -1.8767 -1.00928", "1536 1536"]]
for i in tqdm(range(3)):
    test(testcase[i][0], testcase[i][1], testcase[i][2])
x = list(range(1, 4))
plt.bar([i - 0.15 for i in x], x1, width=0.3, align='center')
plt.bar([i + 0.15 for i in x], x2, width=0.3, color='r', align='center')
plt.xticks(range(1, 4), x)
plt.savefig('parametercompare.png')
plt.show()
