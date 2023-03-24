import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
start = time.time()
end = time.time()
x = []
y = []
for i in tqdm(range(1, 5)):
    for j in tqdm(range(1, 5), leave=False):
        start = time.time()
        os.system(
            f"srun -n{i} -c{j} ./hw2 {j} 0.7725 -0.385 1.3065 0.782 -0.178 0.312 1024 1024 a.png")
        end = time.time()
        x.append(f"n{i} c{j}")
        y.append(end-start)

plt.bar(range(len(y)), y)
plt.xticks(range(len(x)), x, rotation='vertical')
# plt.axis([0, 10, 0, 100])
plt.savefig('timetest.png')
plt.show()
