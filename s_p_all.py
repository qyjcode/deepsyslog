import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import pickle
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
        filename="model_s_p_all/s_p_all.log",
        filemode="a",
        format="%(asctime)s %(name)s:%(levelname)s"
               ":%(filename)s-[line%(lineno)d]-:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def generate(file_name,window_size):
    sentence_vectors=load('embed/s2v f+g sif-fse d=300')
    paramlist = load('data8-2/train_n_param')
    num_sessions = 0
    inputs = []
    labels = []
    num_vec = []
    with open(file_name, 'r',True) as f:
        klist = f.readlines()
        num_sessions = len(klist)
        for i in range(len(klist)):
            line = list(map(lambda n: n - 1, map(int, klist[i].strip().split())))
            for j in range(len(line) - window_size):
                vector = []
                for a in range(j,j+window_size):
                    vector.append(sentence_vectors[line[a]])
                numdata = [0 for k in range(12)]
                for k in range(j+window_size):
                    if line[k] == 5:
                        numdata[1] += int(paramlist[i][k][3])
                    elif line[k] == 7:
                        numdata[2] += int(paramlist[i][k][1])
                    elif line[k] == 8:
                        numdata[3] += int(paramlist[i][k][1])
                    elif line[k] == 9:
                        numdata[4] += int(paramlist[i][k][1])
                    elif line[k] == 10:
                        numdata[5] += int(paramlist[i][k][1])
                    elif line[k] == 14:
                        numdata[7] += int(paramlist[i][k][2])
                        numdata[6] += int(paramlist[i][k][1])
                        numdata[8] += int(paramlist[i][k][3])
                    elif line[k] == 25:
                        numdata[9] += int(paramlist[i][k][2])
                    elif line[k] == 26:
                        numdata[10] += int(paramlist[i][k][2])
                    elif line[k] == 27:
                        numdata[11] += int(paramlist[i][k][2])
                numdata[0] = int(paramlist[i][j+window_size-1][0])
                inputs.append(vector)
                num_vec.append(numdata)
                labels.append(line[j + window_size])
    print('file name:{} session number:{}'.format(file_name, num_sessions))
    print('training data:{}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float),torch.tensor(num_vec,dtype=torch.float),torch.tensor(labels))
    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,numerical_dim,mid_size=100):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc0 = nn.Linear(hidden_size + numerical_dim, mid_size)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(mid_size, num_classes)

    def forward(self, input,num_vec):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = torch.cat((out[:, -1, :], num_vec), dim=1)
        out = self.fc0(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


def train(input_size, hidden_size, num_layers, num_classes,numerical_dim,window_size,batch_size,num_epochs,path,loginfo,imagefile):
    model = Model(input_size, hidden_size, num_layers, num_classes,numerical_dim).to(device)
    seq_dataset = generate('data8-2/hdfs_train',window_size)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    plt.figure()
    x = list(range(num_epochs))
    y=[]

    train_loss = 0
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        if epoch+1>250:
            optimizer.param_groups[0]['lr'] = 1e-6
        elif epoch+1>200:
            optimizer.param_groups[0]['lr'] = 1e-5
        elif epoch+1>100:
            optimizer.param_groups[0]['lr'] = 1e-4
        elif epoch + 1 > 50:
            optimizer.param_groups[0]['lr'] = 0.0005

        train_loss = 0
        t1 = time.time()
        for step, (seq,num_vec, label) in enumerate(dataloader):
            seq = seq.view(-1, window_size,input_size).to(device)
            num_vec = num_vec.view(-1, numerical_dim).to(device)
            output = model(seq,num_vec)
            loss = criterion(output, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        t2 = time.time()
        y.append(train_loss / len(dataloader.dataset))
        print('Epoch [{}/{}],time_cost:{} Train_loss: {:.10f}'.format(epoch + 1, num_epochs, t2 - t1,
                                                                      train_loss / len(dataloader.dataset)))
    torch.save(model.state_dict(),path)
    plt.plot(x,y)
    plt.savefig(imagefile)
    print('Finished Training')
    logging.info(loginfo.format(num_epochs,train_loss / len(dataloader.dataset)))


def readfile(filename,window_size):
    hdfs = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            line = line + [-1] * (window_size + 1 - len(line))
            hdfs.append(tuple(line))
    print('Number of sessions({}): {}'.format(filename, len(hdfs)))
    return hdfs

def get_numdata(line,paramlist):
    numdata = [0 for k in range(12)]
    numdata[0]=paramlist[len(line)-1][0]
    for k in range(len(line)):
        if line[k] == 5:
            numdata[1] += int(paramlist[k][3])
        elif line[k] == 7:
            numdata[2] += int(paramlist[k][1])
        elif line[k] == 8:
            numdata[3] += int(paramlist[k][1])
        elif line[k] == 9:
            numdata[4] += int(paramlist[k][1])
        elif line[k] == 10:
            numdata[5] += int(paramlist[k][1])
        elif line[k] == 14:
            numdata[7] += int(paramlist[k][2])
            numdata[6] += int(paramlist[k][1])
            numdata[8] += int(paramlist[k][3])
        elif line[k] == 25:
            numdata[9] += int(paramlist[k][2])
        elif line[k] == 26:
            numdata[10] += int(paramlist[k][2])
        elif line[k] == 27:
            numdata[11] += int(paramlist[k][2])
    return numdata


def predict(input_size, hidden_size, num_layers, num_classes,numerical_dim,window_size,model_path,num_candidates):
    model = Model(input_size, hidden_size, num_layers, num_classes,numerical_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_normal_data = readfile('data8-2/hdfs_test_normal',window_size)
    test_abnormal_data = readfile('data8-2/hdfs_test_abnormal',window_size)

    normal_paramlist = load('data8-2/test_n_param')
    abnormal_paramlist = load('data8-2/abnormal_param')

    sentence_vectors = load('embed/s2v f+g sif-fse d=300')

    TP =FP = 0
    start_time = time.time()
    with torch.no_grad():
        for i in range(len(test_abnormal_data)):
            line = test_abnormal_data[i]
            for j in range(len(line) - window_size):
                if -1 in line:
                    TP += 1
                    break
                input=[]
                for k in range(j,j+window_size):
                    input.append(sentence_vectors[line[k]])
                numdata=get_numdata(line[0:j+window_size],abnormal_paramlist[i][0:j+window_size])
                label = line[j + window_size]
                input = torch.tensor(input, dtype=torch.float).view(-1, window_size,input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                num_vec = torch.tensor(numdata,dtype=torch.float).view(1,numerical_dim).to(device)
                output = model(input,num_vec)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break
    FN = len(test_abnormal_data) - TP
    print("TP={},FN={}".format(TP, FN))
    print("abnormal test time:", time.time() - start_time)

    with torch.no_grad():
        for i in range(len(test_normal_data)):
            line = test_normal_data[i]
            for j in range(len(line) - window_size):
                input = []
                for k in range(j, j + window_size):
                    input.append(sentence_vectors[line[k]])
                numdata = get_numdata(line[0:j + window_size], normal_paramlist[i][0:j + window_size])
                label = line[j + window_size]
                input = torch.tensor(input, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                num_vec = torch.tensor(numdata, dtype=torch.float).view(1, numerical_dim).to(device)
                output = model(input, num_vec)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break

    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    logging.info('FP: {}, FN: {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('FP: {}, FN: {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('total time: {}'.format(time.time() - start_time))


window_size = 10
input_size = 300
hidden_size = 128
num_layers = 2
num_classes = 29
num_epochs = 300
batch_size = 2048*4
numerical_dim=12

num_candidates=9

model_path1 = 'model/sentence embedding sif.pt'
model_path2='model/sentence_vectors_quick-thought.pt'

loginfo1='sentence embedding sif model changed   epoch={},last train_loss{}'
loginfo2='sentence embedding quick-thought model changed   epoch={}'

for i in range(3):
    model_path = 'model_s_p_all/sentence embedding sif{}.pt'.format(i)
    imagefile='model_s_p_all/trainloss{}'.format(i)
    train(input_size, hidden_size, num_layers, num_classes,numerical_dim,window_size,batch_size,num_epochs,model_path,loginfo1,imagefile)
    predict(input_size, hidden_size, num_layers, num_classes,numerical_dim,window_size,model_path,num_candidates)