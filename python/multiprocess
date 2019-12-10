import multiprocessing as mp

def job(x):
    return x*x

pool = mp.Pool()
res = pool.map(job, range(10))

with mp.Pool(processes=10) as pool:
    res = pool.map(job, range(10))
    print(res)
