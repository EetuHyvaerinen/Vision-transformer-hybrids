import matplotlib.pyplot as plt

def plot_predictions(batch_test, preds, labels_map, offset=0):
    fig = plt.figure(figsize=(8, 8))
    for i in range(16):
        pred = preds[i]
        label = batch_test[1][offset+i]
        pred_txt = labels_map[int(pred)]
        label_txt = labels_map[int(label)]
        
        correct = pred == label
        ax = fig.add_subplot(4, 4, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        img = batch_test[0][offset+i].view(28, 28)
        ax.imshow(img, cmap='gray_r')
        
        ax.text(0.25, 1.1, '{}'.format(label_txt), 
                size=8, color='green',
                horizontalalignment='center',
                verticalalignment='center', 
                transform=ax.transAxes)
        ax.text(0.75, 1.1, '{}'.format(pred_txt), 
                size=8, color='blue' if correct else 'red',
                horizontalalignment='center',
                verticalalignment='center', 
                transform=ax.transAxes)
        ax.text(0.45, 0.05, '{}'.format('Correct' if correct else 'Incorrect'), 
            size=10, color='blue' if correct else 'red',
            horizontalalignment='center',
            verticalalignment='center', 
            transform=ax.transAxes)

    plt.show()