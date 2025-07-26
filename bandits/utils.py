from typing import List
import matplotlib.pyplot as plt


def mk_clear_dir(d: str, delete_existing: bool) -> str:
    import os, shutil
    if os.path.exists(d) and delete_existing:
        shutil.rmtree(d)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def box_plots(data: List):
    fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(data)

    plt.grid()

    # show plot
    plt.show()
