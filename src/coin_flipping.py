import numpy as np
import matplotlib.pyplot as plt

class CoinFlipSimulator():
    def __init__(self, p = 0.5):
        self.p = p
        self.n = 0
        self.heads = 0
        self.tails = 0
    
    def flip(self):
        self.n += 1
        if np.random.random() < self.p:
            self.heads += 1
            return "H"
        else:
            self.tails += 1
            return "T"
        
    def get_results(self):
        return {
            "heads": self.heads,
            "tails": self.tails,
            "n": self.n
        }
    
    def visualize(self):
        plt.bar(["Heads", "Tails"], [self.heads, self.tails], color=["blue", "red"])
        plt.title("Coin Flip Results (n = {})".format(self.n))

        width = 0.35
        x = np.arange(2)
        fig, ax = plt.subplots()
        rects1 = ax.bar(0, self.heads, label='Heads', color='blue')
        rects2 = ax.bar(1, self.tails, label='Tails', color='red')

        ax.set_ylabel('Count')
        ax.set_title('Coin Results (n = {})'.format(self.n))
        ax.set_xticks(x)
        ax.set_xticklabels(["Heads", "Tails"])
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