import numpy as np
import matplotlib.pyplot as plt

class BanditSimulator():
    """
    A class to simulate a multi-armed bandit problem.
    """

    def __init__(self, p = [0.7, 0.5, 0.3]):
        self.p = p
        self.k = len(p)
        self.n = 0
        self.successes = [0] * self.k
        self.failures = [0] * self.k

    def pull(self, i):
        self.n += 1
        if np.random.random() < self.p[i]:
            self.successes[i] += 1
            return 1
        else:
            self.failures[i] += 1
            return 0
        
    def get_results(self):
        return {
            "successes": self.successes,
            "failures": self.failures,
            "n": self.n
        }

    def visualize(self):
        x = np.arange(self.k)
        width = 0.35

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, self.successes, width, label='Successes', color='blue')
        rects2 = ax.bar(x + width/2, self.failures, width, label='Failures', color='red')

        ax.set_xlabel('Machine')
        ax.set_ylabel('Count')
        ax.set_title('Bandit Results (n = {})'.format(self.n))
        ax.set_xticks(x)
        ax.set_xticklabels([f"Machine {i+1}" for i in range(self.k)])
        ax.legend()

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        return plt