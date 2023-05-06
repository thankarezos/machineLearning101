from matplotlib.animation import FuncAnimation
from sklearn.exceptions import ConvergenceWarning
import createDataset as cd
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import plots as pl


# X = cd.linearSeparated(504)
X = cd.nonLinearCenter(504)
# X = cd.nonLinearXOR(504)
# X = cd.nonLinear(504)
X_train, y_train, X_test, y_test = cd.splitData(X)

clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=0)
warnings.filterwarnings("ignore", category=ConvergenceWarning)



fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
# anim2 = pl.fit_plot2(clf, X_train, y_train, X_test, y_test, axs[0,1], fig)
# anim3 = pl.fit_plot3(clf, X_train, y_train, X_test, y_test, axs[1,0], fig)
# anim4 = pl.fit_plot4(clf, X_train, y_train, X_test, y_test, axs[1,1], fig)
anim = pl.animation(clf, X_train, y_train, X_test, y_test, axs, fig)



plt.show()


    
