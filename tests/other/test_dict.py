import time

n = 256
a = {}

st = time.time_ns()
for i in range(n):
    a[i] = i
ed = time.time_ns()

print((ed - st) / 1e3, "us")

st = time.time_ns()

a[999] = 1

ed = time.time_ns()
print((ed - st) / 1e3, "us")