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

    def forward(self, batchData):
        # 0 images:torch.Size([8, 4, 13, 3, 200, 100])
        # 1 bboxes:torch.Size([8, 4, 13, 4])
        # 2 actions:torch.Size([8, 4, 13])
        # 3 activities:torch.Size([8, 4])
        # 4 bboxesNum:torch.Size([8, 4])
        # 5 imgs:torch.Size([8, 4, 3, 480, 720])
        batchSize = batchData[0].shape[0]
        numFrames = batchData[0].shape[1]

        # 传入模型images，bboxes，bboxesNum，imgs
        actionsScores, activitiesScores = self.model(batchData[0], batchData[1], batchData[4])

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
        self.calculateLoss(batchSize, actionsScores, activitiesScores, actionsIn, activitiesIn)
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

    # =======================Val Begin================== #
    @torch.no_grad()
    def val(self):
        model = self.model
        if self.config.trainOrVal is 'val':
            epoch = self.loadModel(self.config.checkPointPath)
            print("epoch:", epoch)
        model.eval()
        self.losses = AverageMeter()
        self.actionsMeter = AverageMeter()
        self.activitiesMeter = AverageMeter()
        self.activitiesPerclassMeter = [AverageMeter() for i in range(self.config.activitiesNumClasses)]
        for batchData in self.valLoader:
            # batchData是一个list中包含tensor的数据需要通过for循环，将每一个tensor加入GPU中
            batchData = [b.to(device=self.config.device) for b in batchData]
            # 0 images:torch.Size([8, 4, 13, 3, 200, 100])
            # 1 bboxes:torch.Size([8, 4, 13, 4])
            # 2 actions:torch.Size([8, 4, 13])
            # 3 activities:torch.Size([8, 4])
            # 4 bboxesNum:torch.Size([8, 4])
            # 5 imgs:torch.Size([8, 4, 3, 480, 720])
            batchSize = batchData[0].shape[0]
            numFrames = batchData[0].shape[1]
            # 传入模型images，bboxes，bboxesNum
            actionsScores, groupScores, activitiesScores, glb = self.model(batchData[0], batchData[1], batchData[4],
                                                                           batchData[6], batchData[7])
            groupIn = self.bulidGroupLabel(glb).cuda()
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
            self.calculateLoss(batchSize, actionsScores, groupScores, activitiesScores, actionsIn, activitiesIn,
                               groupIn)
            # 计算准确率
            self.calculateAccuracy(actionsScores, activitiesScores, actionsIn, activitiesIn)
        activitiesPerclassList = []
        for i in range(self.config.activitiesNumClasses):
            activitiesPerclassList.append(self.activitiesPerclassMeter[i].avg * 100)

        # valResult = {'epoch': self.epoch, 'activitiesAcc': self.activitiesMeter.avg * 100}
        valResult = {
            'epoch': self.epoch,
            'loss': self.losses.avg,
            'activitiesAcc': self.activitiesMeter.avg * 100,
            'actionsAcc': self.actionsMeter.avg * 100,
            'activitiesPerclassAcc': activitiesPerclassList
        }
        print('Epoch: [%d] '
              'Val Activity Accuracy %.3f%%\t'
              % (self.epoch, self.activitiesMeter.avg * 100), "activitiesPerclassAcc:",
              activitiesPerclassList)
        return valResult

    # =======================Val End================== #

    # =======================Save Model Begin================== #

    # =======================Save Model End================== #

    # =======================Build Model,Optimizer,Dataset,Criterion Begin================== #

    def build(self):
        self.buildSeed()
        self.buildModel()
        self.buildDatasetLoader()  # 加载数据集
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
