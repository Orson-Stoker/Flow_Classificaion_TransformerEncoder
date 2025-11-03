from model import *
from packet_extractor import *
from packet_dataset import *

def train(model,trainset,testset,batch_size,learning_rate=0.01,epoch=10,gpu=True):

   
    total_train_step=0  
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    train_loader=DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=testset,batch_size=batch_size,shuffle=True)
    
    if gpu==True:
        model=model.cuda()
        loss_fn=loss_fn.cuda()


    test_loss=[]
    test_acc=[]
    for i in range(epoch):
        print("------------ training turn {}-------------".format(i+1))
        model.train()
        for data in train_loader:
            sequences,targets=data
            if gpu==True:
                sequences=sequences.cuda()
                targets=targets.cuda()

            outputs=model(sequences)
            loss= loss_fn(outputs,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step=total_train_step+1
            if total_train_step%100 == 0:
                print(f"training steps:{total_train_step}, loss:{loss}")

        total_test_loss=0
        total_accuracy=0

        model.eval()
        with torch.no_grad():
            for data in test_loader:
                sequences,targets = data
                if gpu==True:
                    sequences=sequences.cuda()
                    targets=targets.cuda()
                    
                outputs=model(sequences)
                loss = loss_fn(outputs,targets)
                accuracy=(outputs.argmax(1)==targets).sum()

                total_test_loss=total_test_loss+ loss
                total_accuracy=total_accuracy +accuracy

        print(f"loss in test dataset : {round(float(total_test_loss/len(testset)*65),5)}")
        print(f"accuracy in test dataset : {round(float(total_accuracy/len(testset)),5)}")


if __name__ == "__main__":
    # pcap_files=[r"data\0.pcap"]
    # labels=[0]

    # testset=trainset=PacketSequenceDataset(pcap_files,labels,100)

    # model=FlowClassifier(config_file="config.json")
    # train(model,trainset,testset,32)

