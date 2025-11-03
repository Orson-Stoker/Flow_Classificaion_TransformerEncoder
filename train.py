from model import *
from packet_extractor import *
from packet_dataset import *
from sklearn.model_selection import StratifiedKFold
import csv,datetime

class Trainer:
    def __init__(self,config_file):
        with open(config_file, 'r') as f:
            config= json.load(f)["training"]
        self.classifier=FlowClassifier(config_file)
        self.dataset=PacketSequenceDataset(config['pcap_dir'],config['max_flows'])
        
        self.learning_rate=config["learning_rate"]
        self.batch_size=config['batch_size']
        self.epochs=config['epochs']
        self.gpu=config['gpu']
        self.log_steps=config['log_steps']
        self.splits=config['splits']
        self.log_file=config['logging_path']
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "epoch", "test_loss", "test_acc", "timestamp"])



    def log(self, fold, epoch, test_loss, test_acc):        
        timestamp = datetime.datetime.now().isoformat()
    
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([fold, epoch, test_loss, test_acc, timestamp])
    
    def train(self,fold,trainset,testset):
        total_train_step=0

        loss_fn=nn.CrossEntropyLoss()
        optimizer=torch.optim.Adam(self.classifier.parameters(),lr=self.learning_rate)
    
        train_loader=DataLoader(dataset=trainset,batch_size=self.batch_size,shuffle=True)
        test_loader=DataLoader(dataset=testset,batch_size=self.batch_size,shuffle=True)
    
        if self.gpu==1:
            self.classifier=self.classifier.cuda()
            loss_fn=loss_fn.cuda()
        
        for epoch in range(self.epochs):
            print("------------ training turn {}-------------".format(epoch+1))
            self.classifier.train()
            for data in train_loader:
                sequences,targets=data
                if self.gpu==1:
                    sequences=sequences.cuda()
                    targets=targets.cuda()

                outputs=self.classifier(sequences)
                loss= loss_fn(outputs,targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_step=total_train_step+1
                if total_train_step%self.log_steps == 0:
                    print(f"training steps:{total_train_step}, loss:{loss}")

        total_test_loss=0
        total_accuracy=0

        self.classifier.eval()
        with torch.no_grad():
            for data in test_loader:
                sequences,targets = data
                if self.gpu==True:
                    sequences=sequences.cuda()
                    targets=targets.cuda()
                    
                outputs=self.classifier(sequences)
                loss = loss_fn(outputs,targets)
                accuracy=(outputs.argmax(1)==targets).sum()

                total_test_loss=total_test_loss+ loss
                total_accuracy=total_accuracy +accuracy

            print(f"loss in test dataset : {round(float(total_test_loss/len(testset)*65),5)}")
            print(f"accuracy in test dataset : {round(float(total_accuracy/len(testset)),5)}")
            self.log(fold,epoch,round(float(total_test_loss/len(testset)*65)),round(float(total_accuracy/len(testset)),5))

    def cross_valid(self):

        x = self.dataset.sequences
        y = self.dataset.labels
        kfold = StratifiedKFold(n_splits=self.splits, shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(kfold.split(x,y)):

            print(f'======**Fold{fold + 1} Training Begin...**======')
            train_subset = Subset(self.dataset, train_idx)
            test_subset = Subset(self.dataset, test_idx)
            self.train(fold,train_subset,test_subset)




if __name__ == "__main__":
    
    trainer=Trainer("config.json")
    trainer.cross_valid()
