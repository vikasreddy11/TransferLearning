import torch 
import torchvision

#setting
Batch=32
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Epochs=10
train_accs,val_accs=[],[]
train_losses,val_losses=[],[]


#Data loading
train_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ColorJitter(
        brightness=0.2,contrast=0.2,saturation=0.2),
    torchvision.transforms.RandomCrop(224,padding=4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5,0.5,0.5],   
                                     [0.5,0.5,0.5])
])

val_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5,0.5,0.5],
                                     [0.5,0.5,0.5])
])

train_datasets=torchvision.datasets.ImageFolder(root='cat-and-dog/training_set',transform=train_transforms)
val_datasets=torchvision.datasets.ImageFolder(root='cat-and-dog/test_set',transform=val_transforms)

train_loader=torch.utils.data.DataLoader(dataset=train_datasets,shuffle=True,batch_size=Batch)
val_loader=torch.utils.data.DataLoader(dataset=val_datasets,shuffle=False,batch_size=Batch)

#buid model
class build_CNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(build_CNN,self).__init__(*args, **kwargs)

        self.cnn1=torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.cnn2=torch.nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.cnn3=torch.nn.Conv2d(64,128,kernel_size=3,padding=1)

        self.bn1=torch.nn.BatchNorm2d(32)
        self.bn2=torch.nn.BatchNorm2d(64)
        self.bn3=torch.nn.BatchNorm2d(128)

        self.pool=torch.nn.MaxPool2d(2,2)

        self.fn1=torch.nn.Linear(128*28*28,512)
        self.fn2=torch.nn.Linear(512,128)
        self.fn3=torch.nn.Linear(128,2)

        self.relu=torch.nn.ReLU()
        self.Dropout=torch.nn.Dropout(0.4)

    def forward(self,x):

        x=self.cnn1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.pool(x)
        

        x=self.cnn2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.cnn3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.pool(x)

        x=x.view((-1,28*28*128))

        x=self.fn1(x)
        x=self.relu(x)
        x=self.Dropout(x)

        x=self.fn2(x)
        x=self.relu(x)
        x=self.Dropout(x)

        x=self.fn3(x)
        return x

model=build_CNN()
model=model.to(DEVICE)

#optimizer and loss
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
#train
best_val_acc=0
for epoch in range(Epochs):
    model.train()
    running_loss,total,correct=0.0,0,0
    for images,labels in train_loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        predicted=outputs.argmax(1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
        running_loss+=loss.item()

    train_acc=(correct*100)/total
    train_loss=running_loss/len(train_loader)

    with torch.no_grad():
        model.eval()
        running_loss,total,correct=0.0,0,0
        for images,labels in val_loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            outputs=model(images)
            loss=criterion(outputs,labels)

            predicted=outputs.argmax(1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)
            running_loss+=loss.item()

        val_acc=(correct*100)/total
        val_loss=running_loss/len(val_loader)
    
    if val_acc>best_val_acc:
        best_val_acc=val_acc
    
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    val_accs.append(val_acc)
    val_losses.append(val_loss)

    print(f'Epoch{epoch+1}/{Epochs}')
    print(f"Training Accuracy {train_acc:.2f}")
    print(f"Test accuracy: {val_acc:.2f}")

print(f"Best Accuracy :{best_val_acc:.2f}")

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def training_plot(train_accs,val_accs,train_losses,val_losses):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    ax1.plot(train_accs,label='Train')
    ax1.plot(val_accs,label='Val')
    ax1.set_title("Accuracy")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('%')
    ax1.legend()

    ax2.plot(train_losses,label='Train')
    ax2.plot(val_losses,label='Val')
    ax2.set_title("Loss")
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('conv_train.png')
    plt.show()
    print("Saved figure")

#confusion matrix
def plot_confusion_matrix(model,val_loader,class_name):
    all_preds,all_labels=[],[]
    model.eval()
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            outputs=model(images)
            preds=outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    cm=confusion_matrix(all_labels,all_preds)
    plt.figure(figsize=(12,8))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                    xticklabels=class_name,
                    yticklabels=class_name)

    plt.title("Confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("True")
    plt.savefig("convConfusion.png")
    plt.show()
    print("Confusion matrix is saved")

#sample predictions
def plot_predicted(model,val_loader,class_name,n=8):
    model.eval()
    images,labels=next(iter(val_loader))
    with torch.no_grad():
        images=images.to(DEVICE)
        preds=model(images).argmax(1).cpu().numpy()

        mean=torch.tensor([0.5,0.5,0.5]).view(3,1,1).to(DEVICE)
        std=torch.tensor([0.5,0.5,0.5]).view(3,1,1).to(DEVICE)
        images=(images*std+mean).clamp(0,1).cpu()

        fig,axes=plt.subplots(1,n,figsize=(2*n,3))
        for i,ax in enumerate(axes):
            ax.imshow(images[i].permute(1,2,0))
            color='green' if preds[i]==labels[i] else 'red'
            ax.set_title(f'P : {class_name[preds[i]]}\n'
                         f'T : {class_name[labels[i]]}',
                         color=color,fontsize=8)
            ax.axis('off')

        plt.savefig('cnnpredicted.png')
        plt.show()
        print('Saved predicted image')

class_name=train_datasets.classes
training_plot(train_accs,val_accs,train_losses,val_losses)
plot_confusion_matrix(model,val_loader,class_name)
plot_predicted(model,val_loader,class_name)