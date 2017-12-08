import multiprocessing as mp

def printlog(*msg):
    print('['+mp.current_process().name+']', *msg)