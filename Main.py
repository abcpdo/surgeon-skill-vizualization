import classifier
#import predictor
from dataset import JigsawsDataset
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import os
classifier_pretrained = False
predictor_pretrained = False
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Create dataloaders 
combined_dataset = JigsawsDataset(['NoviceSamplesG6.csv','ExpertSamplesG6.csv'])
combined_dataloader = DataLoader(combined_dataset, batch_size=10, shuffle=True, num_workers=0)
novice_dataset = JigsawsDataset(['NoviceSamplesG6.csv'])
novice_dataloader = DataLoader(novice_dataset, batch_size=10, shuffle=True, num_workers=0)
expert_dataset = JigsawsDataset(['ExpertSamplesG6.csv'])
expert_dataloader = DataLoader(expert_dataset, batch_size=10, shuffle=True, num_workers=0)

predict_novice_dataset = PredictionDataset(novice_dataset[:]['X'],40)
predict_novice_dataloader = DataLoader(predict_novice_dataset,batch_size=10,shuffle=True,num_workers=0)

# train or load classifier network
if(not pretrained):
    all_accs = list()
    end_accs = list()

    for i in trange(1): # random cross validation
        train_dataset, test_dataset = random_split(combined_dataset, [int(combined_dataset.__len__()*0.7), int(combined_dataset.__len__())-int(combined_dataset.__len__()*0.7)], generator=torch.Generator().manual_seed(i+10))
        train_dataloader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False, num_workers=0)
        model = classifier.LSTMClassifier(train_dataset[:]['X'].size(2), 30) #input dim, hidden dim
        model, train_accs, test_accs = classifier.train_model(model, train_dataloader, test_dataloader, train_dataset, test_dataset, 100) # model, train loader, test loader, train set, test set, epochs
        acc = classifier.model_accuracy(model, test_dataset[:]['X'], test_dataset[:]['y'], True)
        end_accs.append(acc.item())
        all_accs.append([train_accs, test_accs])
    classifier.plot(end_accs, all_accs)

    # save trained model
    PATH = __location__ + '/LSTM.pth'
    torch.save(model.state_dict(), PATH)
    PATH = __location__ + '/fig.png'
    plt.savefig(PATH, dpi=250)
else:
    PATH = __location__ + '/LSTM.pth'
    model = classifier.LSTMClassifier(combined_dataset[:]['X'].size(2), 30) #input dim, hidden dim
    model.load_state_dict(torch.load(PATH))
    model.eval()
    test_novice_sample = novice_dataset[2]['X']
    pred = model.forward(test_novice_sample.unsqueeze(0),1)
    Sigmoid = nn.Sigmoid()
    pred = torch.max(Sigmoid(pred.detach()),0)[1]
    print(pred.item())

# predictor


# #take novice sample
# predict_novice()

