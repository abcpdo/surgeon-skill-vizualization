import classifier
import predictor
from dataset import *
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

if torch.cuda.is_available(): 
    dev = "cuda:0" 
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Using CUDA")
    torch.cuda.empty_cache()
else:
    dev = "cpu"
    print("Using CPU")  
device = torch.device(dev)

classifier_pretrained = True
predictor_novice_pretrained = True
predictor_expert_pretrained = False

combined_dataset = JigsawsDataset(['NoviceSamplesG6.csv','ExpertSamplesG6.csv'],ratio = 10)
combined_dataloader = DataLoader(combined_dataset, batch_size=10, shuffle=True, num_workers=0)
novice_dataset = JigsawsDataset(['NoviceSamplesG6.csv'],ratio = 10)
#novice_dataloader = DataLoader(novice_dataset, batch_size=10, shuffle=True, num_workers=0)
expert_dataset = JigsawsDataset(['ExpertSamplesG6.csv'], ratio = 10)
#expert_dataloader = DataLoader(expert_dataset, batch_size=10, shuffle=True, num_workers=0)

# train or load classifier network
if not classifier_pretrained:
    train_epochs = 120
    all_accs = list()
    end_accs = list()
    for i in trange(5): # random cross validation
        train_dataset, test_dataset = random_split(combined_dataset, [int(combined_dataset.__len__()*0.7), int(combined_dataset.__len__())-int(combined_dataset.__len__()*0.7)], generator=torch.Generator().manual_seed(i+10))
        train_dataloader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False, num_workers=0)
        model = classifier.LSTMClassifier(train_dataset[:]['X'].size(2), hidden_dim=30) #input dim, hidden dim
        model, train_accs, test_accs = classifier.train_model(model, train_dataloader, test_dataloader, train_dataset, test_dataset, train_epochs) # model, train loader, test loader, train set, test set, epochs
        acc = classifier.model_accuracy(model, test_dataset[:]['X'], test_dataset[:]['y'], True)
        end_accs.append(acc.item())
        all_accs.append([train_accs, test_accs])
    classifier.plot(end_accs, all_accs)

    # save trained model
    PATH = __location__ + '/Classifier.pth'
    torch.save(model.state_dict(), PATH)
    PATH = __location__ + '/acc.png'
    plt.savefig(PATH, dpi=250)   
else:
    PATH = __location__ + '/Classifier.pth'
    model = classifier.LSTMClassifier(combined_dataset[:]['X'].size(2), 30) # input dim, hidden dim
    model.load_state_dict(torch.load(PATH))
    model.eval()
    test_novice_sample = novice_dataset[2]['X']
    pred = model.forward(test_novice_sample.unsqueeze(0).cuda(), 1)
    Sigmoid = nn.Sigmoid()
    pred = torch.max(Sigmoid(pred.detach()), 0)[1]
    #classifier.model_accuracy()

# Sequence Prediction
predict_test_index = -1
input_length = 1  #seconds
input_window = input_length*30   #30hz*time
future_window = 20
train_epochs = 1000
hidden = 1000
offset = 20
stride = 1
input_trunc_seconds = 0 #seconds
input_trunc = input_trunc_seconds*30

input = expert_dataset[predict_test_index]['X'].detach()
input_seq = input[input.sum(axis=1) != 0]
input = input_seq[offset:offset+input_window-input_trunc,:]

# novice predictor
predict_novice_dataset = PredictionDataset(novice_dataset[:]['X'], input_window, stride)
predict_novice_dataloader = DataLoader(predict_novice_dataset, batch_size=1000, shuffle=True, num_workers=0,pin_memory=True)

if not predictor_novice_pretrained:
    predict_novice_model = predictor.LSTMPredictor(predict_novice_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_novice_dataset[:]['X'].size(2))
    predict_novice_model.to(device)
    predict_novice_model = predictor.train_model(predict_novice_model,predict_novice_dataloader,epochs=train_epochs,stride=stride)
    PATH = __location__ + '/Predictor_N.pth'
    torch.save(predict_novice_model.state_dict(), PATH)
    predict_novice_model.eval()
    future_N = predictor.predict(predict_novice_model, input.cuda(), window = input_window,future=future_window,stride=stride)
else:
    PATH = __location__ + '/Predictor_N.pth'
    predict_novice_model = predictor.LSTMPredictor(predict_novice_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_novice_dataset[:]['X'].size(2))
    predict_novice_model.load_state_dict(torch.load(PATH))
    predict_novice_model.to(device)
    predict_novice_model.eval()
    future_N = predictor.predict(predict_novice_model, input.cuda(), window = input_window,future=future_window,stride=stride)

# expert predictor
predict_expert_dataset = PredictionDataset(expert_dataset[:-2]['X'], input_window, stride)
predict_expert_dataloader = DataLoader(predict_expert_dataset, batch_size=1000, shuffle=True, num_workers=0,pin_memory=True)

if not predictor_expert_pretrained:
    predict_expert_model = predictor.LSTMPredictor(predict_expert_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_expert_dataset[:]['X'].size(2))
    predict_expert_model.to(device)
    predict_expert_model = predictor.train_model(predict_expert_model,predict_expert_dataloader,epochs=train_epochs,stride=stride)
    PATH = __location__ + '/Predictor_E.pth'
    torch.save(predict_expert_model.state_dict(), PATH)
    predict_expert_model.eval()
    future_E = predictor.predict(predict_expert_model, input.cuda(), window=input_window,future=future_window,stride=stride)
else:
    PATH = __location__ + '/Predictor_E.pth'
    predict_expert_model = predictor.LSTMPredictor(predict_expert_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_expert_dataset[:]['X'].size(2))
    predict_expert_model.load_state_dict(torch.load(PATH))
    predict_expert_model.to(device)
    predict_expert_model.eval()
    future_E = predictor.predict(predict_expert_model, input.cuda(), window=input_window,future=future_window,stride=stride)

# plotting
input_seq = input_seq.cuda()

fig = plt.figure(figsize=plt.figaspect(0.5))
plt.title('Predicting Gesture') 
ax = fig.add_subplot(1,2,1, projection='3d')
xline1 = input[:, 0]
yline1 = input[:, 1]
zline1 = input[:, 2]
ax.plot3D(xline1, yline1, zline1, 'gray',label="Input")
xline2 = torch.cat((input_seq[offset+input_window-1:offset+input_window,0],future_N[:,0].detach()),0)
yline2 = torch.cat((input_seq[offset+input_window-1:offset+input_window,1],future_N[:,1].detach()),0)
zline2 = torch.cat((input_seq[offset+input_window-1:offset+input_window,2],future_N[:,2].detach()),0)
#ax.plot3D(xline2.cpu(), yline2.cpu(), zline2.cpu(), 'red',label="Novice")
xline3 = torch.cat((input_seq[offset+input_window-1:offset+input_window,0],future_E[:,0].detach()),0)
yline3 = torch.cat((input_seq[offset+input_window-1:offset+input_window,1],future_E[:,1].detach()),0)
zline3 = torch.cat((input_seq[offset+input_window-1:offset+input_window,2],future_E[:,2].detach()),0)
ax.plot3D(xline3.cpu(), yline3.cpu(), zline3.cpu(), 'green',label="Expert")
xline4 = input_seq[offset+input_window-1:offset+input_window+future_window,0]
yline4 = input_seq[offset+input_window-1:offset+input_window+future_window,1]
zline4 = input_seq[offset+input_window-1:offset+input_window+future_window,2]
ax.plot3D(xline4.cpu(), yline4.cpu(), zline4.cpu(), 'blue',linestyle = 'dotted',label="Ground Truth")
ax.legend()
ax.title.set_text('Translation')

ax2 = fig.add_subplot(1,2,2, projection='3d')
xline1 = input[:,3]
yline1 = input[:,4]
zline1 = input[:,5]
ax2.plot3D(xline1, yline1, zline1, 'gray',label="Input")
xline2 = torch.cat((input_seq[offset+input_window-1:offset+input_window,3], future_N[:,3].detach()),0)
yline2 = torch.cat((input_seq[offset+input_window-1:offset+input_window,4], future_N[:,4].detach()),0)
zline2 = torch.cat((input_seq[offset+input_window-1:offset+input_window,5], future_N[:,5].detach()),0)
#ax2.plot3D(xline2.cpu(), yline2.cpu(), zline2.cpu(), 'red',label="Novice")
xline3 = torch.cat((input_seq[offset+input_window-1:offset+input_window,3], future_E[:,3].detach()),0)
yline3 = torch.cat((input_seq[offset+input_window-1:offset+input_window,4], future_E[:,4].detach()),0)
zline3 = torch.cat((input_seq[offset+input_window-1:offset+input_window,5], future_E[:,5].detach()),0)
ax2.plot3D(xline3.cpu(), yline3.cpu(), zline3.cpu(), 'green',label="Expert")
xline4 = input_seq[offset+input_window-1:offset+input_window+future_window,3]
yline4 = input_seq[offset+input_window-1:offset+input_window+future_window,4]
zline4 = input_seq[offset+input_window-1:offset+input_window+future_window,5]
ax2.plot3D(xline4.cpu(), yline4.cpu(), zline4.cpu(), 'blue',linestyle = 'dotted', label="Ground Truth")
ax2.legend()
ax2.title.set_text('Rotation (Euler Angles)')
plt.show()