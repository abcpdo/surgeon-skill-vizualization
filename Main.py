import classifier
import predictor
from dataset import JigsawsDataset, PredictionDataset
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import os
classifier_pretrained = True
predictor_novice_pretrained = True
predictor_expert_pretrained = True
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


combined_dataset = JigsawsDataset(['NoviceSamplesG6.csv','ExpertSamplesG6.csv'])
combined_dataloader = DataLoader(combined_dataset, batch_size=10, shuffle=True, num_workers=0)
novice_dataset = JigsawsDataset(['NoviceSamplesG6.csv'])
#novice_dataloader = DataLoader(novice_dataset, batch_size=10, shuffle=True, num_workers=0)
expert_dataset = JigsawsDataset(['ExpertSamplesG6.csv'])
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
    pred = model.forward(test_novice_sample.unsqueeze(0), 1)
    Sigmoid = nn.Sigmoid()
    pred = torch.max(Sigmoid(pred.detach()), 0)[1]


# Sequence Prediction
predict_test_index = 14
sliding_window = 50
future_window = 20
train_epochs = 40
hidden = 100
offset = 20
stride = 1

# novice predictor
predict_novice_dataset = PredictionDataset(novice_dataset[:]['X'],sliding_window)
predict_novice_dataloader = DataLoader(predict_novice_dataset, batch_size=3000, shuffle=True, num_workers=0)
if not predictor_novice_pretrained:
    predict_novice_model = predictor.LSTMPredictor(predict_novice_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_novice_dataset[:]['X'].size(2))
    predict_novice_model = predictor.train_model(predict_novice_model,predict_novice_dataloader,epochs=train_epochs)
    PATH = __location__ + '/Predictor_N.pth'
    torch.save(predict_novice_model.state_dict(), PATH)
else:
    PATH = __location__ + '/Predictor_N.pth'
    predict_novice_model = predictor.LSTMPredictor(predict_novice_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_novice_dataset[:]['X'].size(2))
    predict_novice_model.load_state_dict(torch.load(PATH))
    predict_novice_model.eval()
    input = novice_dataset[predict_test_index]['X']
    input = input[input.sum(dim=1)!=0]
    input = input[offset:sliding_window+offset,:]
    future_N = predictor.predict(predict_novice_model, input, window=sliding_window,future=future_window)

# expert predictor
predict_expert_dataset = PredictionDataset(expert_dataset[:]['X'],sliding_window)
predict_expert_dataloader = DataLoader(predict_expert_dataset, batch_size=3000, shuffle=True, num_workers=0)

if not predictor_novice_pretrained:
    predict_expert_model = predictor.LSTMPredictor(predict_expert_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_expert_dataset[:]['X'].size(2))
    predict_expert_model = predictor.train_model(predict_expert_model,predict_expert_dataloader,epochs=train_epochs)
    PATH = __location__ + '/Predictor_E.pth'
    torch.save(predict_expert_model.state_dict(), PATH)
else:
    PATH = __location__ + '/Predictor_E.pth'
    predict_expert_model = predictor.LSTMPredictor(predict_expert_dataset[:]['X'].size(2),hidden_dim=hidden,output_dim=predict_expert_dataset[:]['X'].size(2))
    predict_expert_model.load_state_dict(torch.load(PATH))
    predict_expert_model.eval()
    input = novice_dataset[predict_test_index]['X']
    input_seq = input[input.sum(dim=1)!=0]
    input = input_seq[offset:sliding_window+offset,:]
    future_E = predictor.predict(predict_expert_model, input, window=sliding_window,future=future_window)

# plotting
fig = plt.figure(figsize=plt.figaspect(0.5))
plt.title('Predicting Gesture') 
ax = fig.add_subplot(1,2,1, projection='3d')
xline1 = input[:,0].detach().numpy()
yline1 = input[:,1].detach().numpy()
zline1 = input[:,2].detach().numpy()
ax.plot3D(xline1, yline1, zline1, 'gray',label="Input")
xline2 = future_N[:,0].detach().numpy()
yline2 = future_N[:,1].detach().numpy()
zline2 = future_N[:,2].detach().numpy()
ax.plot3D(xline2, yline2, zline2, 'red',label="Novice")
xline3 = future_E[:,0].detach().numpy()
yline3 = future_E[:,1].detach().numpy()
zline3 = future_E[:,2].detach().numpy()
ax.plot3D(xline3, yline3, zline3, 'green',label="Expert")
xline4 = input_seq[offset+sliding_window:offset+sliding_window+future_window,0].detach().numpy()
yline4 = input_seq[offset+sliding_window:offset+sliding_window+future_window,1].detach().numpy()
zline4 = input_seq[offset+sliding_window:offset+sliding_window+future_window,2].detach().numpy()
ax.plot3D(xline4, yline4, zline4, 'blue',label="Ground Truth")
ax.legend()
ax.title.set_text('Translation')

ax2 = fig.add_subplot(1,2,2, projection='3d')
xline1 = input[:,3].detach().numpy()
yline1 = input[:,4].detach().numpy()
zline1 = input[:,5].detach().numpy()
ax2.plot3D(xline1, yline1, zline1, 'gray',label="Input")
xline2 = future_N[:,3].detach().numpy()
yline2 = future_N[:,4].detach().numpy()
zline2 = future_N[:,5].detach().numpy()
ax2.plot3D(xline2, yline2, zline2, 'red',label="Novice")
xline3 = future_E[:,3].detach().numpy()
yline3 = future_E[:,4].detach().numpy()
zline3 = future_E[:,5].detach().numpy()
ax2.plot3D(xline3, yline3, zline3, 'green',label="Expert")
xline4 = input_seq[offset+sliding_window:offset+sliding_window+future_window,3].detach().numpy()
yline4 = input_seq[offset+sliding_window:offset+sliding_window+future_window,4].detach().numpy()
zline4 = input_seq[offset+sliding_window:offset+sliding_window+future_window,5].detach().numpy()
ax2.plot3D(xline4, yline4, zline4, 'blue',label="Ground Truth")
ax2.legend()
ax2.title.set_text('Rotation (Euler Angles)')
plt.show()