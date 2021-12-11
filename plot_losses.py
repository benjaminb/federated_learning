import matplotlib.pyplot as plt


def plot_losses(losses):
    plt.plot(losses, color="c")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig('loss.png')