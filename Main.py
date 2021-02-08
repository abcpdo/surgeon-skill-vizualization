from classifier import LSTMClassifier, train_model, model_accuracy
#from predictor import *
from dataset import JigsawsDataset
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import torch
import statistics as st
pretrained = False

# Create dataloaders 
combined_dataset = JigsawsDataset(['NoviceSamplesG4.csv','ExpertSamplesG4.csv'])
combined_dataloader = DataLoader(combined_dataset, batch_size=10, shuffle=True, num_workers=0)
novice_dataset = JigsawsDataset(['NoviceSamplesG4.csv'])
novice_dataloader = DataLoader(novice_dataset, batch_size=10, shuffle=True, num_workers=0)
expert_dataset = JigsawsDataset(['ExpertSamplesG4.csv'])
expert_dataloader = DataLoader(expert_dataset, batch_size=10, shuffle=True, num_workers=0)


# train classifier network
if(not pretrained):
    all_accs = list()
    end_accs = list()

    for i in trange(5): # random cross validation
        train_dataset, test_dataset = random_split(combined_dataset, [int(combined_dataset.__len__()*0.7), int(combined_dataset.__len__())-int(combined_dataset.__len__()*0.7)], generator=torch.Generator().manual_seed(i+10))
        train_dataloader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False, num_workers=0)

        model = LSTMClassifier(train_dataset[:]['X'].size(2), 30) #input dim, hidden dim
        model, train_accs, test_accs = train_model(model, train_dataloader, test_dataloader, train_dataset, test_dataset, 100) # model, train loader, test loader, train set, test set, epochs
        acc = model_accuracy(model,test_dataset[:]['X'],test_dataset[:]['y'],True)
        end_accs.append(acc.item())
        all_accs.append([train_accs, test_accs])
    
    print(end_accs)
    print("Mean: {}".format(st.stdev(end_accs)))

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    all_accs = np.array(all_accs)
    plt.plot(np.transpose(np.mean(all_accs[:,0], axis = 0)),label='Train', color = (0.5+np.random.random()*0.5,np.random.random()*0.1,np.random.random()*0.1))
    plt.plot(np.transpose(np.mean(all_accs[:,1], axis = 0)),label='Test', color = (np.random.random()*0.1,0.5+np.random.random()*0.5,np.random.random()*0.1))
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)
    plt.title('LSTM Expert/Novice Classifier')
    axis = plt.gca()
    axis.set_ylim(axis.get_ylim()[::-1])
    plt.show()

    # save trained model
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    PATH = __location__ + '/LSTM.pth'
    torch.save(model.state_dict(), PATH)
    #PATH = __location__ + '/fig.png'
    #plt.savefig(PATH,dpi=250)


# classification 

# #take novice sample
# predict_novice()

