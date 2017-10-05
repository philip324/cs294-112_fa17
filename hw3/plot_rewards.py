import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle

prob1 = False

if __name__ == '__main__':
    if prob1:
        with open("saved_rewards/batch32.pkl", "rb") as handle32:
            batch32 = pickle.load(handle32)
        time_step, mean_batch32, best_batch32 = batch32["time"], batch32["mean"], batch32["best"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        mean32, = ax.plot(time_step, mean_batch32, label="mean rew for batch size 32")
        best32, = ax.plot(time_step, best_batch32, label="best rew for batch size 32")
        ax.set_title("Performance with Default Setting")
        ax.set_xlabel("time step")
        ax.set_ylabel("episode reward")
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2g'))
        ax.legend(handles=[mean32, best32], loc="lower right", prop={'size': 11})
        plt.show()
    else:
        # batch size = 8, 16, 32, 64
        with open("saved_rewards/batch8.pkl", "rb") as handle8:
            batch8 = pickle.load(handle8)
        with open("saved_rewards/batch16.pkl", "rb") as handle16:
            batch16 = pickle.load(handle16)
        with open("saved_rewards/batch32.pkl", "rb") as handle32:
            batch32 = pickle.load(handle32)
        with open("saved_rewards/batch64.pkl", "rb") as handle64:
            batch64 = pickle.load(handle64)

        time_step = batch32["time"]
        mean_batch8, best_batch8 = batch8["mean"], batch8["best"]
        mean_batch16, best_batch16 = batch16["mean"], batch16["best"]
        mean_batch32, best_batch32 = batch32["mean"], batch32["best"]
        mean_batch64, best_batch64 = batch64["mean"], batch64["best"]

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
        ax.set_title("Performance with Different Batch Size")
        ax.set_xlabel("time step")
        ax.set_ylabel("episode reward")
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2g'))
        lst = [mean8, best8, mean16, best16, mean32, best32, mean64, best64]
        ax.legend(handles=lst, loc="best", prop={'size': 11})
        plt.show()

