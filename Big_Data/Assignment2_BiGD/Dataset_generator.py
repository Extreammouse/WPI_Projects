import random
import csv

num_points = 5000
num_clusters = random.randint(1, 500)
std_y = 2.0
mu_y  = 2.5
std_z = 1
mu_z  = 0

x = [random.normalvariate(0, 1) for _ in range(num_points)]
y = [random.normalvariate(mu_y, std_y) for _ in range(num_points)]
z = [random.normalvariate(mu_z, std_z) for _ in range(num_points)]

listd = list(zip(x, y, z))

with open('/Users/ehushubhamshaw/Desktop/large_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z'])
    writer.writerows(listd)