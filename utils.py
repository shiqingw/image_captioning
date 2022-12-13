import matplotlib.pyplot as plt
import pickle
from matplotlib import rcParams
from textwrap import wrap

def save_image(img, full_path, caption=None):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(10, 10), dpi=100, frameon=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)

    if caption is not None:
        title = ax.set_title("\n".join(wrap(caption, 45)), fontsize=30)

    #unnormalize 
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224 
    img[2] = img[2] * 0.225 
    img[0] += 0.485 
    img[1] += 0.456 
    img[2] += 0.406
    
    img = img.numpy().transpose((1, 2, 0))
    
    ax.imshow(img)
    ax.axis('off')
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    plt.savefig(full_path, dpi = 100)
    plt.close(fig)
    return


def save_dict(dict_obj, fullname):
    with open(fullname, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(fullname):
    with open(fullname, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    return loaded_obj

def plot_loss(train_loss, validation_loss, test_loss, loss_path):
    fig = plt.figure(figsize=(10, 6), dpi=100, frameon=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    ax.set_xlabel("epoch", fontsize=50)
    ax.set_ylabel("loss", fontsize=50)
    ax.plot(train_loss, linewidth = 5, label="training loss")
    ax.plot(validation_loss, linewidth = 5, label="validation loss")
    ax.plot(test_loss, linewidth = 5, label="testing loss")
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.grid()
    ax.legend(fontsize = 40)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=100)
    plt.close(fig)

def plot_bleu_scores(bleu_dict, loss_path):
    fig = plt.figure(figsize=(10, 6), dpi=100, frameon=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    ax.set_xlabel("epoch", fontsize=50)
    ax.set_ylabel("bleu scores", fontsize=50)
    ax.plot(bleu_dict["bleu-1"], linewidth = 5, label="bleu-1")
    ax.plot(bleu_dict["bleu-2"], linewidth = 5, label="bleu-2")
    ax.plot(bleu_dict["bleu-3"], linewidth = 5, label="bleu-3")
    ax.plot(bleu_dict["bleu-4"], linewidth = 5, label="bleu-4")
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.grid()
    ax.legend(fontsize = 40)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=100)
    plt.close(fig)


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f