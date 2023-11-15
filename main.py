from data_process import *
from models import *
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

class Animator:  
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid()

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # plt.show()
        
#训练函数
def train(epochs):
	model.train() #模型设置成训练模式
	animator = Animator(xlabel='epoch', xlim=[1, epochs], ylim=[0.01, 1.0], legend=['train loss', 'train acc'])
	for epoch in range(epochs): 	#训练epochs轮
		total_acc, total_count = 0, 0  #记录每轮acc
		loss_sum = 0  				#记录每轮loss
		for batch in train_iter:
			input_, aspect, label = batch
			optimizer.zero_grad() #每次迭代前设置grad为0

			#不同的模型输入不同，请同学们看model.py文件
			# output = model(input_)
			output = model(input_, aspect)

			loss = criterion(output, label) #计算loss
			loss.backward() #反向传播
			optimizer.step() #更新模型参数
			total_acc += (output.argmax(1) == label).sum().item() #累计正确预测数
			total_count += label.size(0) #累积总数
			loss_sum += loss.item() #累积loss
		animator.add(epoch + 1, [loss_sum / len(train_iter), total_acc / total_count])
		print('epoch: ', epoch, 'loss: ', loss_sum / len(train_iter), 'acc: ', total_acc / total_count)
	
	plt.show()
	test_acc = evaluate() #模型训练完后进行测试
	print('test_acc:', test_acc)

#测试函数
def evaluate():
	model.eval()
	total_acc, total_count = 0, 0
	loss_sum = 0
	all_preds, all_labels = [], []

	with torch.no_grad(): #测试时不计算梯度
		for batch in test_iter:
			input_, aspect, label = batch

			# predicted_label = model(input_)
			predicted_label = model(input_, aspect)

			loss = criterion(predicted_label, label) #计算loss
			total_acc += (predicted_label.argmax(1) == label).sum().item() #累计正确预测数
			total_count += label.size(0) #累积总数
			loss_sum += loss.item() #累积loss
   
			all_preds.extend(predicted_label.argmax(1).cpu().numpy())
			all_labels.extend(label.cpu().numpy())
		print('test_loss:', loss_sum / len(test_iter))

	conf_matrix = confusion_matrix(all_labels, all_preds)
    # Plot confusion matrix
	plot_confusion_matrix(conf_matrix, classes=['negative', 'positive', 'neutral'], title='Confusion Matrix')
	return total_acc/total_count

# plot confusion matrix
def plot_confusion_matrix(conf_matrix, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

TORCH_SEED = 21 #随机数种子
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #设置模型在几号GPU上跑
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #设置device

# 设置随机数种子，保证结果一致
os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#创建数据集
train_dataset = MyDataset('./data/acsa_train.json')
test_dataset = MyDataset('./data/acsa_test.json')
train_iter = DataLoader(train_dataset, batch_size=25, shuffle=True, collate_fn=batch_process)
test_iter = DataLoader(test_dataset, batch_size=25, shuffle=False, collate_fn=batch_process)

# 加载我们的Embedding矩阵
embedding = torch.tensor(np.load('./emb/my_embeddings.npz')['embeddings'], dtype=torch.float)


#定义模型
# model = BiLSTM_Model(embedding).to(device)
# model = AEBiLSTM_Model(embedding).to(device)
model = AEBiLSTMWithSoftAttention_Model(embedding).to(device)
# model = AEBiLSTMWithSelfAttention_Model(embedding).to(device)

#定义loss函数、优化器
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.001)

#开始训练
train(100)
