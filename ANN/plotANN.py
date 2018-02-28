import matplotlib.pyplot as plt


dataset = [0.64461899, 0.6381194, 0.63071918, 0.62305468, 0.61522949, 0.6072374, 0.59930474, 0.591424525, 0.58345842, 0.57543129, 0.56689054, 0.55877626, 0.55110556, 0.54350519, 0.53531247, 0.52739054, 0.51917452, 0.51103848, 0.5034582, 0.49572641]
y = range(20)
plt.plot(y,dataset)
plt.ylabel('weights for CYT')
plt.xlabel('number of epochs')
plt.axis([0, 20,.4,.7])

plt.show()