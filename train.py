import torchvision.datasets as datasets
import matplotlib.pyplot as plt

from utils import *

dataset = datasets.ImageFolder('./dataset', transform=data_transform)

#split train/val : 80/20
train_set, val_set = torch.utils.data.random_split(dataset, [2208, 552])

print('Train set:', len(train_set))
print('Validation set:', len(val_set))

#Load dataset
batch_size = 32
train_load = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_load = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

#Show img after load
# def imgshow(img):
#     img = img/2 + 0.5 
#     np_img = img.numpy()
#     plt.figure(figsize=(20, 20))
#     plt.imshow(np.transpose(np_img, (1, 2, 0)))
#     plt.show()

# data_iter = iter(val_load)
# img, labels = data_iter.next()
# imgshow(torchvision.utils.make_grid(img))

model = CNN()

train_loss = []
val_loss = []
train_acc = []
val_acc = []

def Training_Model(model, epochs, parameters):
    #Using CrossEntropyLoss, optim SGD
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(parameters, lr=0.01)

    model = model.cuda()
    
    for epoch in range(epochs): 
        start = time.time()
        correct = 0
        iterations = 0
        iter_loss = 0.0
        
        model.train() #Set mode Train                  
        
        for i, (inputs, labels) in enumerate(train_load, 0):
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            #Convert to Cuda() to use GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()  
            
            #Forward
            outputs = model(inputs)
            
            #Calculating loss
            loss = loss_f(outputs, labels)  
            iter_loss += loss.item()
            
            #Backpropagation
            loss.backward()              
            optimizer.step()             
            
            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1
    

        train_loss.append(iter_loss/iterations)
        train_acc.append((100 * correct / len(train_set)))
   

        #Eval on validation set
        loss = 0.0
        correct = 0
        iterations = 0

        model.eval() #Set mode evaluation

        #No_grad on Val_set
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_load, 0):
                
                inputs = Variable(inputs)
                labels = Variable(labels)
                
                #To Cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                #Forward and Caculating loss
                outputs = model(inputs)     
                loss = loss_f(outputs, labels) 
                loss += loss.item()

                # Record the correct predictions for val data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1

            val_loss.append(loss/iterations)
            val_acc.append((100 * correct / len(val_set)))

        stop = time.time()
        
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Val Loss: {:.3f}, Val Accuracy: {:.3f}, Time: {}s'
            .format(epoch+1, epochs, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1],stop-start))

epochs = 32
Training_Model(model=model, epochs=epochs, parameters=model.parameters())

#Save model
torch.save(model.state_dict(), 'weights/Face-Mask-Model.pt')

#Show chart acc and save Acc_chart
plt.plot(train_acc, label='Train_Accuracy')
plt.plot(val_acc, label='Val_Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.axis('equal')
plt.legend(loc=7)
plt.savefig('Acc_chart.png')
plt.show()
