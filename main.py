import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor

from helper import Dataset
from model import SafetyNet
import numpy as np
from pytorch_model_summary import summary
#from torchmetrics.functional import precision_recall, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split

VALIDATION_SET = None

def main():
    global VALIDATION_SET
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Instantiating NN
    net = SafetyNet()
    net.train()
    net = net.float().to(device)


    # Start dataset loading
    trainset = Dataset(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    #x_train, x_test, y_train, y_test = train_test_split(trainset,trainset.labels,test_size=0.2, stratify=trainset.labels)

    testset = Dataset(root_dir=path_data,
                       indice_dir=path_indices,
                       mode='test',
                       size=SIZE,
                       transform=transforms.Compose([ToTensor()]))

    testset, VALIDATION_SET = torch.utils.data.random_split(testset, [int(testset.data.shape[0] * 0.5), int(testset.data.shape[0] * 0.5) +1])

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    #testset = Dataset(root_dir=path_data,
    #                       indice_dir=path_indices,
     #                      mode='test',
     #                      size=SIZE,
     #                      transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    Train_prec = []
    Train_recall = []
    Train_beta_score = []

    Test_loss = []
    Test_acc = []
    Test_prec = []
    Test_recall = []
    Test_beta_score = []
    model_int8 = None

    print("Start training")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)

        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        precision = [0,0,0]
        recall = [0,0,0]
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['safety'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            try:
                running_loss += loss.item()
            except:
                print(labels)
            i += 1
            #print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f Train Precisoin: %.5f Train Recall: %.5f' %
                  #(epoch + 1, i, accuracy / i, running_loss / i, precision[0]/i, recall[0]/i))

        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
            (epoch + 1, i, accuracy / i, running_loss / i))

        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0

        for data_test in testloader:
            net.eval()


            IEGM_test, labels_test = data_test['safety'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()


            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print('Test Acc: %.5f Test Loss: %.5f ' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())


    torch.save(net, './saved_models/Safety_net_{0}.pkl'.format(0))

#    torch.save(model_int8, './saved_models/IEGM_net_quant{0}.pkl'.format(model_number))

    torch.save(net.state_dict(), './saved_models/Safety_net_state_dict_{0}.pkl'.format(0))

    n = np.zeros(shape=(1, 1, 63, 1), dtype=np.float32)
    sumry = summary(net, torch.tensor(n).to(device))
    print(sumry)

    file = open('./saved_models/model_{0}_info.txt'.format(0), 'a')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))

    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    file.write('\n\n')
    file.write(sumry)
    file.write('\n\n')
    file.close()

    print('Finish training')

    #return Test_acc[len(Test_acc)-1], Test_beta_score[len(Test_beta_score)-1]

def evaluate():
    BATCH_SIZE_TEST = args.batchsz

    net = torch.load('saved_models/Safety_net_0.pkl')
    net = net.float().to(device)

    validLoader = DataLoader(VALIDATION_SET, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)
    correct = 0.0
    total = 0.0
    i = 0.0
    running_loss_test = 0.0
    label_list = []
    predict_list = []
    for data_test in validLoader:
        net.eval()

        IEGM_test, labels_test = data_test['safety'], data_test['label']
        IEGM_test = IEGM_test.float().to(device)
        labels_test = labels_test.to(device)
        outputs_test = net(IEGM_test)
        _, predicted_test = torch.max(outputs_test.data, 1)
        total += labels_test.size(0)
        correct += (predicted_test == labels_test).sum()

        label_list = label_list + labels_test.tolist()
        predict_list = predict_list + predicted_test.tolist()
        i += 1


    print('Test Acc: %.5f' % (correct / total))

    classes = ('unknown', 'serious', 'minor', 'possible','fatal', 'no-injury')
    cf_matrix = confusion_matrix(label_list, predict_list)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    print(cf_matrix)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batch sz for traindb', default=100)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=63)
    argparser.add_argument('--path_data', type=str, default='')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()

    device = 'cpu'

    print("device is --------------", device)

    main()
    evaluate()

