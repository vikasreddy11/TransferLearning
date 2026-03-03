import torch
import torchvision

#setting
Architecture='resnet50'
Batch=32
Epoch=10
Num_classes=2
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_accs,train_losses=[],[]
val_accs,val_losses=[],[]


#dataloading
train_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ColorJitter(
        brightness=0.2,contrast=0.2,saturation=0.2),
    torchvision.transforms.RandomCrop(224,padding=4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
])

val_transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485,0.456,0.406],
                                     [0.229,0.224,0.225])
])


train_datasets=torchvision.datasets.ImageFolder(root='cat-and-dog/training_set',transform=train_transforms)
val_datasets=torchvision.datasets.ImageFolder(root='cat-and-dog/test_set',transform=val_transforms)

train_loader=torch.utils.data.DataLoader(dataset=train_datasets,shuffle=True,batch_size=Batch)
val_loader=torch.utils.data.DataLoader(dataset=val_datasets,shuffle=False,batch_size=Batch)

#build model
def buid_model(architecture,num_classes):

    #resnet50
    if architecture=='resnet50':
        model=torchvision.models.resnet50(weights='IMAGENET1K_V2')
        #freeze layers
        for param in model.parameters():
            param.requires_grad=False

        in_features=model.fc.in_features
        model.fc=torch.nn.Sequential(
            torch.nn.Linear(in_features,512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512,num_classes)
        )

    elif architecture=='vgg16':
        model=torchvision.models.vgg16(weights='IMAGENET1K_V1')
        for param in model.features.parameters():
            param.requires_grad=False
        
        in_features=model.classifier[6].in_features
        model.classifier[6]=torch.nn.Sequential(
            torch.nn.Linear(in_features,512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512,num_classes)
        )

    elif architecture=='mobilenet_v2':
        model=torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        for param in model.features.parameters():
            param.requires_grad=False

        in_features=model.classifier[1].in_features
        model.classifier[1]=torch.nn.Sequential(
            torch.nn.Linear(in_features,256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256,num_classes)
        )

    else:
        raise ValueError(f"Invalid architecture{architecture}")
    
    return model

model=buid_model(Architecture,Num_classes)
model=model.to(DEVICE)


#optimizer ,loss and scheduler
optimizer=torch.optim.Adam(
    filter(lambda p: p.requires_grad,model.parameters()),lr=1e-3
)
criterion=torch.nn.CrossEntropyLoss()
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2,
)

#train
best_acc=0
for epoch in range(Epoch):
    model.train()
    running_loss,correct,total=0.0,0,0

    for images,labels in train_loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        predicted=outputs.argmax(1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)

    train_acc=(correct/total)*100
    train_loss=running_loss/(len(train_loader))


    model.eval()
    total, correct = 0, 0
    running_losss  = 0.0 
    with torch.no_grad():
        for images,labels in val_loader:
            images,labels=images.to(DEVICE),labels.to(DEVICE)
            outputs=model(images)
            loss=criterion(outputs,labels)

            running_loss+=loss.item()
            predicted=outputs.argmax(1)
            correct+=(predicted==labels).sum().item()
            total+=labels.size(0)

    val_acc=(correct/total)*100
    val_loss=running_loss/len(val_loader)

    scheduler.step(val_acc)

    if val_acc>best_acc:
        best_acc=val_acc

    train_accs.append(train_acc)
    train_losses.append(train_loss)
    val_accs.append(val_acc)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{Epoch}")
    print(f"Trainning accuracy{train_acc:.2f}%")
    print(f"Test accuracy{val_acc:.2f}%")

print(f"Best accuracy:{best_acc:.2f}")

#plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#trainning cruve
def plot_cruve(train_accs,val_accs,train_losses,val_losses):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    ax1.plot(train_accs,label="Train")
    ax1.plot(val_accs,label="Val")
    ax1.set_title('Accuracy')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("%")
    ax1.legend()

    ax2.plot(train_losses,label="Train")
    ax2.plot(val_losses,label="Val")
    ax2.set_title("loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('feature_E_training.png')
    plt.show()
    print("Saved training curves")

#confusion matrix
def plot_cruves_matrix(model,val_loader,class_names):
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
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusin matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("featureConfusion.png")
    plt.show()
    print("Confusion is saved")

#sample predictions
def plot_predicted(model,val_loader,class_names,n=8):
    model.eval()
    images,labels=next(iter(val_loader))
    with torch.no_grad():
        images=images.to(DEVICE)
        preds=model(images).argmax(1).cpu().numpy()

        mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(DEVICE)
        std=torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(DEVICE)
        images=(images*std+mean).clamp(0,1).cpu()

        fig,axes=plt.subplots(1,n,figsize=(2*n,3))
        for i,ax in enumerate(axes):
            ax.imshow(images[i].permute(1,2,0))
            color='green' if preds[i]==labels[i] else 'red'
            ax.set_title(f'P:{class_names[preds[i]]}\n'
                         f'T:{class_names[labels[i]]}',
                         color=color,fontsize=8)
            ax.axis('off')

        plt.savefig("predicted_feature.png")
        plt.show()
        print("Saved figure")


class_names =train_datasets.classes


plot_cruve(train_accs, val_accs, train_losses, val_losses)
plot_cruves_matrix(model, val_loader, class_names)
plot_predicted(model, val_loader, class_names)