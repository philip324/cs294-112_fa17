import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle

if __name__ == '__main__':
    # batch size = 8, 16, 32, 64
    with open("saved_rewards/batch8.pkl", "rb") as handle8:
        batch8 = pickle.load(handle8)
    with open("saved_rewards/batch16.pkl", "rb") as handle16:
        batch16 = pickle.load(handle16)
    with open("saved_rewards/batch64.pkl", "rb") as handle64:
        batch64 = pickle.load(handle64)

    time_step = batch8["time"]
    mean_batch8, best_batch8 = batch8["mean"], batch8["best"]
    mean_batch16, best_batch16 = batch16["mean"], batch16["best"]
    mean_batch64, best_batch64 = batch64["mean"], batch64["best"]
    
    # batch size = 32
    input_file = os.path.join(os.getcwd(), "reward_batch32.txt")
    assert os.path.exists(input_file), \
            'Path does not exist: {}'.format(input_file)
    with open(input_file) as f:
        entries = [x.strip() for x in f.readlines()]
    mean_batch32, best_batch32 = [],[]
    for i in range(len(entries)):
        idx = entries[i].rfind(" ")
        num = float(entries[i][idx+1:])
        # if i % 3 == 0:
        #     time_step.append(num)
        if i % 3 == 1:
            mean_batch32.append(num)
        elif i % 3 == 2:
            best_batch32.append(num)
    for _ in range(len(time_step) - len(mean_batch32)):
        mean_batch32.append(float('nan'))
        best_batch32.append(float('nan'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    mean8, = ax.plot(time_step, mean_batch8, label="mean rew for batch size 8")
    best8, = ax.plot(time_step, best_batch8, label="best rew for batch size 8")
    mean16, = ax.plot(time_step, mean_batch16, label="mean rew for batch size 16")
    best16, = ax.plot(time_step, best_batch16, label="best rew for batch size 16")
    mean32, = ax.plot(time_step, mean_batch32, label="mean rew for batch size 32")
    best32, = ax.plot(time_step, best_batch32, label="best rew for batch size 32")
    mean64, = ax.plot(time_step, mean_batch64, label="mean rew for batch size 64")
    best64, = ax.plot(time_step, best_batch64, label="best rew for batch size 64")
    ax.set_ylabel("time step")
    ax.set_ylabel("episode reward")
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2g'))
    lst = [mean8, best8, mean16, best16, mean32, best32, mean64, best64]
    ax.legend(handles=lst, loc="best", prop={'size': 10})
    plt.show()

