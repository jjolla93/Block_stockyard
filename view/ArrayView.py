import matplotlib.pyplot as plt
import numpy as np
import simulater.DataManager as dm

def visualize_space(Statuses):
    Days=range(len(Statuses))
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(Statuses)):
        ax = fig.add_subplot(1, len(Statuses)/1+1, i+1)
        #_label=Names[i]+'('+str(Labels[i])+')'
        ax.set_title(Days[i], fontsize='20', x=0.5, y=1.0)
        plt.imshow(Statuses[i], vmin=0, vmax=1)
        #plt.figimage(Statuses[i])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    #ax.set_aspect('equal')
    #색 기준 축 입력
    '''
    cax = fig.add_axes([0.12, 0.1, 0.95, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    #cax.patch.set_alpha(1)
    cax.set_frame_on(False)
    '''
    #plt.colorbar(orientation='vertical')
    plt.show()

def visualize_log(logs, cumulate):
    max_logs = 1
    num_logs = len(logs)
    num_status = len(logs[0]) - 1
    '''
    cumulate = np.zeros([logs[0][0][0].shape[0], logs[0][0][0].shape[1]])
    for i in range(num_logs):
        for j in range(num_status):
            cumulate += logs[i][j][0]
    '''
    if num_logs>max_logs:
        num_logs=max_logs
        logs=logs[0:max_logs]
    fig = plt.figure(figsize=(15, 15))
    for i in range(num_logs):
        for j in range(num_status):
            #ax = fig.add_subplot(num_logs, num_status, num_status*i+j+1)
            ax = fig.add_subplot(max_logs*5, num_status/5, 50*i+j + 1)
            #_label=Names[i]+'('+str(Labels[i])+')'
            ax.set_title(str(logs[i][j][1]), fontsize='7', x=0.5, y=0.88)
            plt.imshow(logs[i][j][0], vmin=-1, vmax=logs[i][j][0].max())
            #plt.figimage(Statuses[i])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    fig.savefig('../data/arrangement.png')
    dm.export_2darray_csv(cumulate, '../data/result.csv')

    print ('Reward of arrangement: ' + str(logs[0][-1]))
    #ax.set_aspect('equal')
    #색 기준 축 입력
    '''
    cax = fig.add_axes([0.12, 0.1, 0.95, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    #cax.patch.set_alpha(1)
    cax.set_frame_on(False)
    '''
    #plt.colorbar(orientation='vertical')
    plt.show()
