import matplotlib.pyplot as plt

def create_plot():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    return ax

def create_plot2():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    return ax

fig, axs = plt.subplots(nrows=1, ncols=2)

# get the axes from the previously created plots
ax1 = create_plot()
axs[0].plot(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata())
plt.close(ax1.figure)

ax2 = create_plot2()
axs[1].plot(ax2.lines[0].get_xdata(), ax2.lines[0].get_ydata())
plt.close(ax2.figure)

# adjust the subplot layout
fig.tight_layout()

plt.show()
