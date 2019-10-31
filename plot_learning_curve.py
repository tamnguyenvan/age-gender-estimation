import pickle as pkl
from matplotlib import pyplot as plt


def visualize(hist):
    plt.subplot(121)
    plt.plot(hist['age_acc'], label='age accuracy')
    plt.plot(hist['gender_acc'], label='gender accuracy')
    plt.plot(hist['val_age_acc'], label='val age accuracy')
    plt.plot(hist['val_gender_acc'], label='val gender accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend(loc='best')

    plt.subplot(122)
    plt.plot(hist['age_loss'], label='age loss')
    plt.plot(hist['gender_loss'], label='gender loss')
    plt.plot(hist['val_age_loss'], label='val age loss')
    plt.plot(hist['val_gender_loss'], label='val gender loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend(loc='best')

    plt.suptitle('Age gender estimation')
    plt.savefig('age_gender_estimation.png')
    plt.show()


hist = pkl.load(open('history.pkl', 'rb'))
visualize(hist)