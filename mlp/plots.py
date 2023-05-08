import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np




def animation(model, X_train, y_train, X_test, y_test, axs, fig, title=None, active=False, callback=None):
    
    epoch = model.max_iter
    
    ax1 = axs[0,0]
    ax2 = axs[0,1]
    ax3 = axs[1,0]
    ax4 = axs[1,1]
    


    #plot 1
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([-0.2, 1])

    ax1.scatter(X_test[:,0], X_test[:,1], c=y_test)


    #plot 2
    ax2.set_xlim([-0.2, 1])
    ax2.set_ylim([-0.2, 1])
    acuracy2 = ax2.text(0.8,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
            transform=ax2.transAxes, ha="center")
    title2 = ax2.text(0.1, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax2.transAxes, ha="center")
    
    x1_empty, x2_empty = np.meshgrid(np.linspace(-0.2, 1, 100), np.linspace(-0.2, 1, 100))
    y_grid_empty = np.empty((100, 100))

    levels = np.linspace(y_grid_empty.min(), y_grid_empty.max(), 11)
    contourSet = ax2.contour(x1_empty, x2_empty, y_grid_empty, levels=levels, colors='k')

    #plot 3

    title3 = ax3.text(0.1, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                    transform=ax3.transAxes, ha="center")
    
    #plot 4
    mse_values = []
    title4 = ax4.text(0.1, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                transform=ax4.transAxes, ha="center")
    line4, = ax4.plot([], [])
    error4 = ax4.text(0.8,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
            transform=ax4.transAxes, ha="center")

    def plot2(epoch):

        y_pred = model.predict(X_test)
        # Predict on X_test

        scatter2 = ax2.scatter(X_test[:,0], X_test[:,1], c=y_pred)


        x1, x2 = np.meshgrid(np.linspace(-0.2, 1, 100), np.linspace(-0.2, 1, 100))
        X_grid = np.vstack((x1.ravel(), x2.ravel())).T

        # Predict the class probabilities for each point on the grid
        y_grid = model.predict_proba(X_grid)[:, 1]

        # Reshape the predicted probabilities into a grid
        y_grid = y_grid.reshape(x1.shape)

        y_min = y_grid.min()
        y_max = y_grid.max()


        if 0.5 >= y_min and 0.5 <= y_max:
            contour2 = ax2.contour(x1, x2, y_grid, levels=[0.5], colors='k')
            
        else:
            contour2 = contourSet

        # Set plot limits and title
        title2.set_text(f'Epoch {epoch + 1}')

        acuracy2.set_text(f'Accuracy: {model.score(X_test, y_test):.3f}')
        return [scatter2, title2, contour2.collections[0], acuracy2]
    
    def plot3(epoch):
        y_pred = model.predict(X_test)

        y_pred_1 = y_pred[y_test == 0]

        y_pred_2 = y_pred[y_test == 1]

        # Create the scatter plot for y_test == 0
        scatter3_1 = ax3.scatter(range(len(y_pred_1)), y_pred_1, marker='x', c='blue', label='y_test == 0')

        # Create the scatter plot for y_test == 1
        scatter3_2 = ax3.scatter(range(len(y_pred_1), len(y_test)), y_pred_2, marker='x', c='red', label='y_test == 1')

        # Set plot limits and title
        title3.set_text(f'Epoch {epoch}')
        return scatter3_1, scatter3_2, title3
    
    def plot4(epoch):
        y_pred = model.predict(X_test)

        mse = np.mean((y_pred - y_test) ** 2)
        mse_values.append(mse)
        ax4.set_xlim([0, model.max_iter])
        ax4.set_ylim([min(mse_values) - min(mse_values) * 0.1 , max(mse_values) + max(mse_values) * 0.1])

        line4.set_data(np.arange(len(mse_values)), mse_values)

        # Set plot limits and title
        title4.set_text(f'Epoch {epoch + 1}')
        error4.set_text(f'Error: {mse:.3f}') 
        return line4, title4, error4
    global best_val_score, n_iter_no_change
    best_val_score = -1
    n_iter_no_change = 0
    

    def update(epoch):
        model.partial_fit(X_train, y_train, classes=[0, 1])
        scatter2, title2, contour2, acuracy2 = plot2(epoch)
        scatter3_1, scatter3_2, title3 = plot3(epoch)
        line4, title4, error4 = plot4(epoch)
        
        

        
        val_score = model.score(X_test, y_test)
        global best_val_score, n_iter_no_change
        
        if val_score > best_val_score:
            best_val_score = val_score
            n_iter_no_change = 0
        else:
            n_iter_no_change += 1
            if n_iter_no_change >= model.n_iter_no_change:
                if callback:
                    callback()
        
        if epoch + 1 == model.max_iter or model.score(X_test, y_test) >= 1:
            if callback:
                callback()
   
        return [scatter2, scatter3_1, scatter3_2, title2, title3, title4, contour2, acuracy2, line4, error4]

    anim = FuncAnimation(fig, update, frames=epoch, blit=True, interval=10, repeat=False )
    return anim

def plot1(model, X_train, y_train, X_test, y_test, ax):
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])
    ax.scatter(X_test[:,0], X_test[:,1], c=y_test)

def plot2(model, X_train, y_train, X_test, y_test, ax):
    y_pred = model.predict(X_test)

    ax.scatter(X_test[:,0], X_test[:,1], c=y_pred)

    x1, x2 = np.meshgrid(np.linspace(-0.2, 1, 100), np.linspace(-0.2, 1, 100))
    X_grid = np.vstack((x1.ravel(), x2.ravel())).T
    
    y_grid = model.predict_proba(X_grid)[:, 1]

    # Reshape the predicted probabilities into a grid
    y_grid = y_grid.reshape(x1.shape)


    ax.contour(x1, x2, y_grid, levels=[0.5], colors='k')

def plot3(model, X_train, y_train, X_test, y_test, ax):
    y_pred = model.predict(X_test)

    y_pred_1 = y_pred[y_test == 0]

    y_pred_2 = y_pred[y_test == 1]

    # Create the scatter plot for y_test == 0
    ax.scatter(range(len(y_pred_1)), y_pred_1, marker='x', c='blue', label='y_test == 0')

    # Create the scatter plot for y_test == 1
    ax.scatter(range(len(y_pred_1), len(y_test)), y_pred_2, marker='x', c='red', label='y_test == 1')
    
def plot4(model, X_train, y_train, X_test, y_test, ax):
    y_pred = model.predict(X_test)
    y_pred_1 = y_pred[y_pred == 0]
    y_pred_2 = y_pred[y_pred == 1]
    y_test_1 = y_test[y_test == 0]
    y_test_2 = y_test[y_test == 1]
    # scatter = ax.scatter(range(len(y_test)), y_test, marker='o', facecolors='none', edgecolors='blue', label='y_test', s=20)
    ax.scatter(range(len(y_test_1)), y_test_1, marker='o', c='blue', label='y_pred == 0', s=50)
    ax.scatter(range(len(y_test_1), len(y_test)), y_test_2, marker='o', c='blue', label='y_pred == 1', s=50)
    ax.scatter(range(len(y_pred_1)), y_pred_1, marker='x', c='red', label='y_test == 0', s=10)
    ax.scatter(range(len(y_pred_1), len(y_pred)), y_pred_2, marker='x', c='red', label='y_test == 1', s=10)