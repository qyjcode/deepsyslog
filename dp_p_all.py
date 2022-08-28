import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import pickle
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    filename="dp/dp_p_all.log",
    filemode="a",
    format="%(asctime)s %(name)s:%(levelname)s"
           ":%(filename)s-[line%(lineno)d]-:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def generate(window_size):
    # load sentence vector
    sv_list = load('sv_list')

    # load sentence vector index
    svIndex = load("svIndex")

    # laod parameter dict
    d = load("numvec_dict")

    # load training data list: [[line indexes for each block, params for each block]]
    train_normal_blocks = load("train_normal_blocks")
    inputs = []
    num_vecs = []
    labels = []

    for block in train_normal_blocks:
        line = block[0]
        param = block[1]
        for j in range(len(line) - window_size):
            vector = []
            num_vec = [0 for k in range(numerical_dim)]
            for a in range(j, j + window_size):
                index = svIndex[line[a]]
                vector.append(sv_list[index])

            for k in range(j + window_size):
                index = svIndex[line[k]]
                dindex = d.get(index)
                if dindex != None:
                    cp = param[k]
                    for pi in range(1, len(cp)):
                        num_vec[dindex] += int(cp[pi])
                        dindex += 1
            num_vec[numerical_dim - 1] = param[j + window_size - 1][0]

            inputs.append(vector)
            num_vecs.append(num_vec)
            labels.append(svIndex[line[j + window_size]])

    print('train block number:{}'.format(len(train_normal_blocks)))
    print('training data length:{}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(num_vecs, dtype=torch.float),
                            torch.tensor(labels))
    return dataset


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, numerical_dim, mid_size=100):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc0 = nn.Linear(hidden_size + numerical_dim, mid_size)
        # self.batch_norm = nn.BatchNorm1d(numerical_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(mid_size, num_classes)

    def forward(self, input, num_vec):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        # num_vec = self.batch_norm(num_vec)
        out = torch.cat((out[:, -1, :], num_vec), dim=1)
        out = self.fc0(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


def train(input_size, hidden_size, num_layers, num_classes, numerical_dim, window_size, batch_size, num_epochs, path,
          loginfo):
    model = Model(input_size, hidden_size, num_layers, num_classes, numerical_dim).to(device)
    seq_dataset = generate(window_size)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_time1 = time.time()

    train_loss = 0
    for epoch in range(num_epochs):
        if epoch + 1 > 250:
            optimizer.param_groups[0]['lr'] = 1e-6
        elif epoch + 1 > 200:
            optimizer.param_groups[0]['lr'] = 1e-5
        elif epoch + 1 > 100:
            optimizer.param_groups[0]['lr'] = 1e-4
        elif epoch + 1 > 50:
            optimizer.param_groups[0]['lr'] = 0.0005

        train_loss = 0
        t1 = time.time()
        for step, (seq, num_vec, label) in enumerate(dataloader):
            seq = seq.view(-1, window_size, input_size).to(device)
            num_vec = num_vec.view(-1, numerical_dim).to(device)
            output = model(seq, num_vec)
            loss = criterion(output, label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        t2 = time.time()
        print('Epoch [{}/{}],time_cost:{} Train_loss: {:.10f}'.format(epoch + 1, num_epochs, t2 - t1,
                                                                      train_loss / len(dataloader.dataset)))
    torch.save(model.state_dict(), path)

    print('Finished Training')
    logging.info(loginfo.format(num_epochs, train_loss / len(dataloader.dataset)))

    train_time2 = time.time()
    logging.info("time cost for training：{}".format(train_time2 - train_time1))


def predict(input_size, hidden_size, num_layers, num_classes, numerical_dim, window_size, model_path, num_candidates):
    model = Model(input_size, hidden_size, num_layers, num_classes, numerical_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sv_list = load('sv_list')
    svIndex = load("svIndex")
    d = load("numvec_dict")
    test_normal_blocks = load("test_normal_blocks")
    abnormal_blocks = load("abnormal_blocks")

    print("normal block number:", len(test_normal_blocks))
    print("abnormal block number:", len(abnormal_blocks))

    test_time1 = time.time()
    test_logs = 0

    TP = FP = 0
    start_time = time.time()
    with torch.no_grad():
        for block in abnormal_blocks:
            line = block[0]
            param = block[1]
            line = line + [-1] * (window_size + 1 - len(line))
            for j in range(len(line) - window_size):
                test_logs += 1
                if -1 in line:
                    TP += 1
                    break
                input = []
                for k in range(j, j + window_size):
                    input.append(sv_list[svIndex[line[k]]])

                num_vec = [0 for k in range(numerical_dim)]
                for k in range(j + window_size):
                    index = svIndex[line[k]]
                    dindex = d.get(index)
                    if dindex != None:
                        cp = param[k]
                        for pi in range(1, len(cp)):
                            num_vec[dindex] += int(cp[pi])
                            dindex += 1
                num_vec[numerical_dim - 1] = param[j + window_size - 1][0]
                label = svIndex[line[j + window_size]]
                input = torch.tensor(input, dtype=torch.float).view(-1, window_size, input_size).to(device)
                num_vec = torch.tensor(num_vec, dtype=torch.float).view(1, numerical_dim).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(input, num_vec)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break
    FN = len(abnormal_blocks) - TP
    print("TP={},FN={}".format(TP, FN))
    print("abnormal test time:", time.time() - start_time)

    with torch.no_grad():
        for block in test_normal_blocks:
            line = block[0]
            param = block[1]
            for j in range(len(line) - window_size):
                test_logs += 1

                input = []
                for k in range(j, j + window_size):
                    input.append(sv_list[svIndex[line[k]]])
                num_vec = [0 for k in range(numerical_dim)]
                for k in range(j + window_size):
                    index = svIndex[line[k]]
                    dindex = d.get(index)
                    if dindex != None:
                        cp = param[k]
                        for pi in range(1, len(cp)):
                            num_vec[dindex] += int(cp[pi])
                            dindex += 1
                num_vec[numerical_dim - 1] = param[j + window_size - 1][0]
                label = svIndex[line[j + window_size]]
                input = torch.tensor(input, dtype=torch.float).view(-1, window_size, input_size).to(device)
                num_vec = torch.tensor(num_vec, dtype=torch.float).view(1, numerical_dim).to(device)
                label = torch.tensor(label).view(-1).to(device)
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

    test_time2 = time.time()
    logging.info("time cost for testing：{}".format(test_time2 - test_time1))
    logging.info(("number of test logs：{}").format(test_logs))


window_size = 6
input_size = 100
hidden_size = 128
num_layers = 2
num_epochs = 100

batch_size = 2048
numerical_dim = 24

num_candidates = 10

loginfo1 = 'model changed epoch={},last train_loss{}'

if __name__ == "__main__":
    logging.info('input_dim={}  hidden_dim={}  num_layer={}  epochs={} batch={} h={} g={}'.
                 format(input_size, hidden_size,
                        num_layers, num_epochs,
                        batch_size, window_size,
                        num_candidates))
    sv_list = load("sv_list")
    print("class_num", len(sv_list))

    model_path = 'dp/dp_p_all h={} layer={} hidden={}.pt'.format(window_size, num_layers, hidden_size)

    train(input_size, hidden_size, num_layers, len(sv_list), numerical_dim, window_size, batch_size, num_epochs,
          model_path, loginfo1)
    predict(input_size, hidden_size, num_layers, len(sv_list), numerical_dim, window_size, model_path, num_candidates)
