from classifier import *
#from predictor import *
from dataset import *
from torch.utils.data import DataLoader
pretrained = False

# Create dataloaders 
combined_dataset = JigsawsDataset(['NoviceSamplesG4.csv','ExpertSamplesG4.csv'])
combined_dataloader = DataLoader(combined_dataset, batch_size=10, shuffle=True, num_workers=0)
novice_dataset = JigsawsDataset(['NoviceSamplesG4.csv'])
novice_dataloader = DataLoader(novice_dataset, batch_size=10, shuffle=True, num_workers=0)



#train classifier network (if not pretrained)
if(not pretrained):
    model = Classifier(train_X.size(2),hidden_dim) #input dim, hidden dim, num_layers, output dim, dropout ratio
    model,train_accs,test_accs = train_model(model,train_X,train_y,epochs) # model, X, y, epochs
    acc = model_accuracy(model,test_X[:,100:200,:],test_y,True)
    accs.append(acc.item())
    all_accs.append([train_accs,test_accs])
    train_network()

# classification 

# #take novice sample
# predict_novice()

