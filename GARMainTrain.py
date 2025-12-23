import shutil
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from util.dataset import returnDataset
from model.GARModel import GARModel
from util.early_stopping import EarlyStopping
from util.showPicture import showAccuracy, showLoss, showTrainPerAccuracy, showTestPerAccuracy
from util.utils import AverageMeter, show_epoch_info, print_log
from torch.utils import data


class GARMain:
    def __init__(self, config):
        self.config = EasyDict(config)
        self.build()

    def saveModel(self, epoch, filepath):
        state = {
            'epoch': epoch,
            # 'conv1x1_state_dict': self.model.conv1x1.state_dict(),
            # 'bn2d_state_dict': self.model.bn2d.state_dict(),
            # 'roi_align_state_dict': self.model.roi_align.state_dict(),
            # 'ln1_state_dict': self.model.ln1.state_dict(),
            # 'fc_state_dict': self.model.fc.state_dict(),
            # 'ln2_state_dict': self.model.ln2.state_dict(),
            # 'spatialEncoder_state_dict': self.model.spatialEncoder.state_dict(),
            # 'selfAttnCon_state_dict': self.model.selfAttnCon.state_dict(),
            # 'temporalEncoder_state_dict': self.model.temporalEncoder.state_dict(),
            # 'dropout_state_dict': self.model.dropout.state_dict(),

            'roi_align_state_dict': self.model.roi_align.state_dict(),
            'bbox_fc_state_dict': self.model.bbox_fc.state_dict(),
            'spatialEncoder_state_dict': self.model.spatialEncoder.state_dict(),
            'selfAttnCon_state_dict': self.model.selfAttnCon.state_dict(),
            'temporalEncoder_state_dict': self.model.temporalEncoder.state_dict(),
            'dropout_state_dict': self.model.dropout.state_dict(),

            # 'actions_fc_state_dict': self.model.actions_fc.state_dict(),
            # 'activities_fc_state_dict': self.model.activities_fc.state_dict(),
            # 'actios_classifier_state_dict': self.model.actions_classifier.state_dict(),
            # 'activities_classifier_state_dict': self.model.activities_classifier.state_dict(),
        }
        torch.save(state, filepath)
        print('model saved to:', filepath)

    # 加载模型
    def loadModel(self, filepath):
        state = torch.load(filepath)
        self.epoch = state['epoch']
        # self.model.conv1x1.load_state_dict(state['conv1x1_state_dict'])
        # self.model.bn2d.load_state_dict(state['bn2d_state_dict'])
        # self.model.roi_align.load_state_dict(state['roi_align_state_dict'])
        # self.model.ln1.load_state_dict(state['ln1_state_dict'])
        # self.model.fc.load_state_dict(state['fc_state_dict'])
        # self.model.ln2.load_state_dict(state['ln2_state_dict'])
        # self.model.spatialEncoder.load_state_dict(state['spatialEncoder_state_dict'])
        # self.model.selfAttnCon.load_state_dict(state['selfAttnCon_state_dict'])
        # self.model.temporalEncoder.load_state_dict(state['temporalEncoder_state_dict'])
        # self.model.dropout.load_state_dict(state['dropout_state_dict'])

        self.model.roi_align.load_state_dict(state['roi_align_state_dict'])
        self.model.bbox_fc.load_state_dict(state['bbox_fc_state_dict'])
        self.model.spatialEncoder.load_state_dict(state['spatialEncoder_state_dict'])
        self.model.selfAttnCon.load_state_dict(state['selfAttnCon_state_dict'])
        self.model.temporalEncoder.load_state_dict(state['temporalEncoder_state_dict'])
        self.model.dropout.load_state_dict(state['dropout_state_dict'])
        print('Load model states from: ', filepath)
        return self.epoch

    # =======================Train Begin================== #
    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def adjust_lr(self, optimizer, new_lr):
        print('change learning rate:', new_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self):
        self.model.train()
        self.model.apply(self.set_bn_eval)
        self.epoch = 1
        # self.config.train.startEpoch = self.loadModel(self.config.checkPointPath)
        train_accuracy = []  # 用于存储训练的准确率
        test_accuracy = []  # 用于存储测试的准确率
        train_loss = []  # 用于存储训练得到的损失值
        test_loss = []  # 用于存储测试得到的损失值
        train_per_accuracy_total = torch.zeros((self.config.activitiesNumClasses, self.config.MAXEpoch))
        # 用于存储训练时每类人群行为的准确率
        test_per_accuracy_total = torch.zeros((self.config.activitiesNumClasses, self.config.MAXEpoch))
        # 用于存储测试时每类人群行为的准确率

        save_path = self.config.resultPath  # 当前目录下
        early_stopping = EarlyStopping(save_path)

        bestResult = {'epoch': 0, 'activitiesAcc': 50, 'loss': 50}
        for epoch in range(self.config.train.startEpoch, self.config.train.totalEpochs):
            # if epoch in self.config.train.lr_plan:  # {41:1e-4, 81:5e-5, 121:1e-5}  改变训练的学习率
            #     self.adjust_lr(self.optimizer, self.config.train.lr_plan[epoch])
            self.lr = self.lr_scheduler.get_lr()[0]
            print("学习率：", self.lr, "epoch:", epoch)  # 显示epoch
            self.epoch = epoch

            trainInfo = self.trainEpoch()

            show_epoch_info('Train', self.config.logPath, trainInfo)  # 显示训练的相关参数信息
            train_accuracy.append(trainInfo['activitiesAcc'])  # 将训练得到的准确率添加到train_accuracy中存储
            train_loss.append(trainInfo['loss'])  # 将训练得到的损失值添加到test_accuracy中存储
            if trainInfo['loss'] < bestResult['loss']:
                bestResult = trainInfo  # 更新最好的准确率
                filepath = self.config.resultPath + '/epoch%d_%.2f%%.pth' % (
                    epoch, trainInfo['activitiesAcc'])  # 模型保存路径
                self.saveModel(self.epoch, filepath)  # 保存模型参数
            # 早停止
            early_stopping(trainInfo['loss'], self.model)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
        # 用于显示经过循环后得到训练和测试的准确率变化图
        showAccuracy(train_accuracy, test_accuracy)
        # 用于显示经过循环后得到训练和测试的损失值变化图
        showLoss(train_loss, test_loss)

    def set_bn_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def trainEpoch(self):
        model = self.model
        self.losses = AverageMeter()
        self.actionsMeter = AverageMeter()
        self.activitiesMeter = AverageMeter()
        self.activitiesPerclassMeter = [AverageMeter() for i in range(self.config.activitiesNumClasses)]

        for batchData in self.trainLoader:
            # train for one epoch
            # batchData是一个list中包含tensor的数据需要通过for循环，将每一个tensor加入GPU中
            batchData = [b.to(device=self.config.device) for b in batchData]
            # 将模型及其子模块中的参数都将设置为可训练状态
            model.train()
            model.apply(self.set_bn_eval)
            # 将batchData传入forward中用来训练
            self.forward(batchData)
            # 用于计算平均值
            self.optimizer.zero_grad()
            self.totalLoss.backward()
            self.optimizer.step()

            # self.losses.update(self.totalLoss.item())
            # self.update()

        activitiesPerclassList = []
        for i in range(self.config.activitiesNumClasses):
            activitiesPerclassList.append(self.activitiesPerclassMeter[i].avg * 100)

        self.lr_scheduler.step()

        train_info = {
            'epoch': self.epoch,
            'loss': self.losses.avg,
            'activitiesAcc': self.activitiesMeter.avg * 100,
            'actionsAcc': self.actionsMeter.avg * 100,
            'activitiesPerclassAcc': activitiesPerclassList
        }
        return train_info

    def bulidGroupLabel(self,tensor_2d):
        # 选择每一行除了-1的第一个元素，并组成一维 tensor
        selected_elements = torch.zeros(tensor_2d.size(0), dtype=tensor_2d.dtype, device=tensor_2d.device)

        for i in range(tensor_2d.size(0)):
            non_minus_one_indices = (tensor_2d[i] != -1).nonzero()
            if non_minus_one_indices.size(0) > 0:
                first_non_minus_one_index = non_minus_one_indices[0].item()
                selected_elements[i] = tensor_2d[i, first_non_minus_one_index]
        return selected_elements

    def forward(self, batchData):
        # 0 images:torch.Size([8, 4, 13, 3, 200, 100])
        # 1 bboxes:torch.Size([8, 4, 13, 4])
        # 2 actions:torch.Size([8, 4, 13])
        # 3 activities:torch.Size([8, 4])
        # 4 bboxesNum:torch.Size([8, 4])
        # 5 imgs:torch.Size([8, 4, 3, 480, 720])
        batchSize = batchData[0].shape[0]
        numFrames = batchData[0].shape[1]
        # images, bboxes, actions, activities, bboxes_num, frames, groupId, groups
        # print(batchData[6].shape)
        # 传入模型images，bboxes，bboxesNum，imgs
        # print(batchData[2])
        # print(batchData[6])
        # print(batchData[7])
        actionsScores, groupScores, activitiesScores, glb = self.model(batchData[0], batchData[1], batchData[4],
                                                                       batchData[6], batchData[7])
        # print("glb",glb)
        groupIn = self.bulidGroupLabel(glb).cuda()
        # print("group7", torch.sum(batchData[7], dim=1), batchData[7].shape)
        # print("group6", torch.sum(batchData[6], dim=1), batchData[6].shape)
        # print("groupScores:", groupScores.shape)
        # actionsIn为行人动作的标签  维度大小为（B, C） B为Batch,C为种类
        # activitiesIn为人群行为的标签  维度大小为(B,T) B为Batch，T为帧数
        # bboxesNum 为行人数量，大小为(B,T) B为Batch，T为帧数
        actionsIn = batchData[2].reshape((batchSize, numFrames, self.config.numMaxN))
        activitiesIn = batchData[3].reshape((batchSize, numFrames))
        bboxesNum = batchData[4].reshape(batchSize, numFrames)

        # 根据行人框数量，获取被行人框选中的行人的动作标签
        actionsInNopad = []
        for b in range(batchSize):
            N = bboxesNum[b][0]  # 行人框数量
            actionsInNopad.append(actionsIn[b][0][:N])
        actionsIn = torch.cat(actionsInNopad, dim=0).reshape(-1, )  # 将被行人框选中的行人的动作标签集合转换成tensor数据
        activitiesIn = activitiesIn[:, 0].reshape(batchSize, )  # 人群行为标签(B,T) 只取T帧中的第一帧的数值，因为T帧中的数值是相同的

        # 运用交叉熵函数计算损失
        self.calculateLoss(batchSize, actionsScores, groupScores, activitiesScores, actionsIn, activitiesIn, groupIn)
        # actionsLoss = self.actionsCriterion(actionsScores, actionsIn)
        # activitiesLoss = self.activitiesCriterion(activitiesScores, activitiesIn)
        # 计算准确率
        self.calculateAccuracy(actionsScores, activitiesScores, actionsIn, activitiesIn)
        # return actionsLoss, activitiesLoss, actionsMeterBatch.avg, activitiesMeterBatch.avg
    # =======================Train End================== #
    def calculateLoss(self, batchSize, actionsScores, groupScores, activitiesScores, actionsIn, activitiesIn, groupIn):
        # 交叉熵损失函数
        # 预测的行人动作结果与行人动作的种类标签做损失
        actionsLoss = self.actionsCriterion(actionsScores, actionsIn)
        # print(groupScores, groupIn)
        groupLoss = self.groupCriterion(groupScores, groupIn.long())
        # 预测的人群行为结果与人群行为的种类标签做损失
        activitiesLoss = self.activitiesCriterion(activitiesScores, activitiesIn)

        # 均方差损失函数
        # 行人均方差
        actionsInOnehot = F.one_hot(actionsIn, self.config.actionsMumClasses)
        actionsMseLoss = F.mse_loss(actionsScores.float(), actionsInOnehot.float())

        groupInOnehot = F.one_hot(groupIn.long(), self.config.groupNumClasses)
        groupMseLoss = F.mse_loss(groupScores.float(), groupInOnehot.float())
        # 人群均方差
        activitiesInOnehot = F.one_hot(activitiesIn, self.config.activitiesNumClasses)
        activitiesMseLoss = F.mse_loss(activitiesScores.float(), activitiesInOnehot.float())

        # 总损失
        # 将行人动作计算的损失值和人群行为计算的损失值相加，作为模型整体的损失值，
        # 其中cfg.actionsLoss_weight作为平衡行人动作损失和人群行为损失的权重

        actionsLossTotal = self.config.actionsLossWeight * (actionsLoss + actionsMseLoss)
        groupLossTotal = self.config.actionsLossWeight * (groupMseLoss + groupLoss)
        activitiesLossTotal = activitiesLoss + activitiesMseLoss
        self.totalLoss = actionsLossTotal + activitiesLossTotal + groupLossTotal

        # self.totalLoss = actionsLoss + activitiesLoss
        self.losses.update(self.totalLoss.item(), batchSize)  # 所有Batch的损失做平均作为最后的损失值

    def calculateAccuracy(self, actionsScores, activitiesScores, actionsIn, activitiesIn):
        # 人群行为识别准确率
        # 计算预测的行人动作预测的准确率
        actionsLabels = torch.argmax(actionsScores, dim=1)  # 返回指定维度最大值的序号
        actionsCorrect = torch.sum(torch.eq(actionsLabels.int(), actionsIn.int()).float())
        actionsAccuracy = actionsCorrect.item() / actionsScores.shape[0]  # 得到准确率
        self.actionsMeter.update(actionsAccuracy, actionsScores.shape[0])  # 所有Batch的准确率做平均作为最后的准确率
        # 计算预测的行人动作预测的准确率
        activitiesLabels = torch.argmax(activitiesScores, dim=1)  # 返回指定维度最大值的序号
        activitiesCorrect = torch.sum(torch.eq(activitiesLabels.int(), activitiesIn.int()).float())
        activitiesAccuracy = activitiesCorrect.item() / activitiesScores.shape[0]  # 得到准确率
        self.activitiesMeter.update(activitiesAccuracy, activitiesScores.shape[0])  # 所有Batch的准确率做平均作为最后的准确率

        # 每类人群行为准确率
        activitiesNum = torch.zeros(self.config.activitiesNumClasses).int()
        activitiesPerclassNum = torch.zeros(self.config.activitiesNumClasses).int()
        for i in range(activitiesLabels.shape[0]):
            activitiesPerclassNum[activitiesIn[i].int()] += 1
            if torch.eq(activitiesLabels[i].int(), activitiesIn[i].int()):
                activitiesNum[activitiesLabels[i].int()] += 1

        # 计算结果
        for i in range(self.config.activitiesNumClasses):
            if activitiesPerclassNum[i] != 0:
                activitiesAccuracy = activitiesNum[i] / activitiesPerclassNum[i]
                self.activitiesPerclassMeter[i].update(activitiesAccuracy, activitiesPerclassNum[i])
            # -----------------------------------每类人群行为准确率-------------------------------#

    # =======================Save Model Begin================== #

    # =======================Save Model End================== #

    # =======================Build Model,Optimizer,Dataset,Criterion Begin================== #
    def update(self):
        self.lr_scheduler.optimizer.step()

    def build(self):
        self.buildSeed()
        self.buildModel()
        self.buildOptimizer()  # 优化器（Optimizer）是用于更新模型参数的算法。优化器的目标是通过最小化损失函数来调整模型的权重，以便在训练过程中逐步提高模型的性能。
        self.buildDatasetLoader()  # 加载数据集
        self.buildScheduler()  # 调度器（Scheduler）是深度学习中用于调整学习率和其他超参数的重要组件之一。它们可以帮助模型在训练过程中动态地调整学习率，以提高训练的效果和稳定性。
        self.buildCriterion()

    def buildSeed(self):
        # np.random.seed(cfg.train_random_seed)  # np.random.seed用于生成指定随机数
        # torch.manual_seed(cfg.train_random_seed)  # 设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的。
        # random.seed(cfg.train_random_seed)  # 用来生成随机数的函数
        seed = self.config.train.seed
        np.random.seed(seed)  # np.random.seed用于生成指定随机数
        torch.manual_seed(seed)  # 设置种子的用意是一旦固定种子，后面依次生成的随机数其实都是固定的。
        random.seed(seed)  # 用来生成随机数的函数

    def buildModel(self):
        model = GARModel(self.config)
        self.model = model.to(self.config.device)

    def buildOptimizer(self):
        config = self.config
        model = self.model
        try:
            optim = getattr(torch.optim, config.train.optimizer.type)
        except Exception:
            raise NotImplementedError('not implemented optim method ' +
                                      config.train.optimizer.type)
        optimizer = optim(filter(lambda p: p.requires_grad, model.parameters()), **config.train.optimizer.kwargs)
        self.optimizer = optimizer

    def buildDatasetLoader(self):
        config = self.config
        trainingSet, validationSet = returnDataset(self.config)
        params = {
            'batch_size': config.dataset.train.batch_size,  # 16
            'shuffle': True,  # 每次取完数据后清洗数据再取
            'num_workers': 4  # 进程个数
        }
        training_loader = data.DataLoader(trainingSet, **params)  # 将训练的dataset中的数据加载到训练的DataLoader中
        params['batch_size'] = config.dataset.val.batch_size  # 设施训练时batch的大小，测试和训练不一样
        validation_loader = data.DataLoader(validationSet, **params)  # 将测试的dataset中的数据加载到测试的DataLoader中
        self.trainLoader = training_loader
        self.valLoader = validation_loader

    # 创建了一个MultiStepLR学习率调度器。学习率调度器用于在训练过程中动态调整学习率。
    # MultiStepLR调度器会在指定的训练步数（milestones）处将学习率乘以一个给定的系数（gamma）。
    # 这些参数都是从配置对象config中获取的。last_epoch参数指定了训练的起始轮数，它的值为self.startEpoch - 1。
    def buildScheduler(self):
        config = self.config
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=config.train.scheduler.milestones,
            gamma=config.train.scheduler.gamma,
            last_epoch=self.config.train.startEpoch - 1)

    # def adjust_lr(self, self.optimizer, new_lr):
    #     print('change learning rate:', new_lr)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = new_lr

    def buildCriterion(self):
        config = self.config
        if config.train.criterion == 'ce_loss':
            actions_weight = torch.tensor(config.train.actions_weight).cuda().detach()
            activities_weight = torch.tensor(config.train.activities_weight).cuda().detach()
            self.actionsCriterion = nn.CrossEntropyLoss(weight=actions_weight)
            self.activitiesCriterion = nn.CrossEntropyLoss(weight=activities_weight)
        else:
            raise NotImplementedError('not implemented criterion ' + config.criterion)
    # =======================Build Model,Optimizer,Dataset,Criterion End================== #
