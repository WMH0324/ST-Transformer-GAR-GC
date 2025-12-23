import numpy as np
import torch
import torch.nn as nn

from util.pytorch_i3d import InceptionI3d
from util.roi_align import RoIAlign
from torchvision.models import resnet50

from util.utils import prep_images
from util.pos_encoding import spatialencoding2d, PositionalEncoding2D
from .GARModelCell import GARModelCell
from .MyMultiHeadAttentionCon import MyMultiheadAttentionCon, ECABlock
from .ResNet50 import MyResNet50
from .Transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder, \
    TransformerEncoderLayerTanhRelu


class GARModel(nn.Module):
    def __init__(self, config):
        super(GARModel, self).__init__()
        # 参数
        self.config = config
        self.embed_features = config.embed_features
        self.spatial_encoding = config.spatial_encoding
        self.numLayers = config.numLayers
        self.KEYT = self.config.KEYT

        # 模型层
        self.resnet = resnet50(pretrained=True)
        self.myresnet = MyResNet50()
        self.conv1x1 = torch.nn.Conv2d(2048, self.embed_features, kernel_size=1)
        self.bn2d = nn.BatchNorm2d(self.embed_features, eps=1e-5, momentum=0.1, affine=True)
        self.ecaBlock = ECABlock(self.embed_features)
        self.roi_align = RoIAlign(config.crop_size[0], config.crop_size[0])
        self.ln1 = nn.LayerNorm([2048, self.config.crop_size[0], self.config.crop_size[0]])
        self.fc = nn.Linear(self.config.crop_size[0] * self.config.crop_size[0] * self.embed_features,
                            self.embed_features)
        self.relu = nn.ReLU()

        self.ln2 = nn.LayerNorm(self.embed_features)
        self.bbox_fc = nn.Sequential(nn.Linear(51200, 1024), nn.ReLU(), nn.Linear(1024, self.embed_features))
        self.selfAttnCon = MyMultiheadAttentionCon(self.embed_features, self.config.headNum,
                                                   dropout=self.config.dropoutNum)
        self.tokenCLSInit = nn.Parameter(torch.randn(1, 1, self.embed_features))

        self.GARModelLayer = nn.ModuleList()
        for i in range(self.numLayers):
            self.GARModelLayer.append(GARModelCell(config, self.embed_features))

        self.dropout = nn.Dropout(p=0.5)
        # 分类器
        self.actions_fc = nn.Linear(self.embed_features, self.config.actionsMumClasses)
        self.activities_fc = nn.Linear(self.embed_features, self.config.activitiesNumClasses)

        self.actions_classifier = nn.Sequential(
            nn.Linear(self.embed_features, self.embed_features * 2),  # 活动类别数 9
            nn.ReLU(),
            nn.Linear(self.embed_features * 2, self.config.actionsMumClasses),
            # nn.Linear(NFG,self.cfg.num_actions),
            nn.Dropout(p=config.dropoutNum),
            nn.Softmax(dim=1)
        )
        self.groupClassifier = nn.Sequential(
            nn.Linear(self.embed_features, self.embed_features * 2),  # 活动类别数 9
            nn.ReLU(),
            nn.Linear(self.embed_features * 2, self.config.groupNumClasses),
            # nn.Linear(NFG,self.cfg.num_activities),
            nn.Dropout(p=config.dropoutNum),
            nn.Softmax(dim=1)
        )
        # 人群分类器groupNumClasses
        self.activities_classifier = nn.Sequential(
            nn.Linear(self.embed_features, self.embed_features * 2),  # 活动类别数 9
            nn.ReLU(),
            nn.Linear(self.embed_features * 2, self.config.activitiesNumClasses),
            # nn.Linear(NFG,self.cfg.num_activities),
            nn.Dropout(p=config.dropoutNum),
            nn.Softmax(dim=1)
        )
        encoderLayer = TransformerEncoderLayer(self.embed_features, self.config.headNum, dropout=self.config.dropoutNum,
                                               normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.spatialEncoder = TransformerEncoder(encoderLayer, num_layers=self.config.encoderLayersNum,
                                                 norm=encoderNorm)

        encoderLayer = TransformerEncoderLayer(self.embed_features, self.config.headNum, dropout=self.config.dropoutNum,
                                               normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.temporalEncoder = TransformerEncoder(encoderLayer, num_layers=self.config.encoderLayersNum,
                                                  norm=encoderNorm)

        encoderLayerG = TransformerEncoderLayerTanhRelu(self.embed_features, self.config.headNum,
                                                        dropout=self.config.dropoutNum,
                                                        normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.spatialGroupEncoder = TransformerEncoder(encoderLayerG, num_layers=self.config.encoderLayersNum,
                                                      norm=encoderNorm)

        encoderLayerG = TransformerEncoderLayer(self.embed_features, self.config.headNum,
                                                dropout=self.config.dropoutNum,
                                                normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.temporalGroupEncoder = TransformerEncoder(encoderLayerG, num_layers=self.config.encoderLayersNum,
                                                       norm=encoderNorm)

        # 解码器
        decoderLayer = TransformerDecoderLayer(self.embed_features, self.config.headNum, dropout=self.config.dropoutNum,
                                               normalize_before=True)
        decoderNorm = nn.LayerNorm(self.embed_features)
        self.garDecoder = TransformerDecoder(decoderLayer, num_layers=self.config.decoderLayersNum, norm=decoderNorm)
        # self.ln = nn.LayerNorm([self.embed_features, self.config.dataset.out_size[0], self.config.dataset.out_size[1]])

    # resnet接口：作为backbone接口
    # 输入是多批次的帧序列
    # 输出是多批次的每一帧的特征图
    def backboneFunc(self, images_in):
        B, T, C, H, W = images_in.shape
        imgsResNet50 = torch.reshape(images_in, (B * T, C, H, W))  # B*T, 3, H, W
        imgsResNet50 = prep_images(imgsResNet50)
        imgsFeatures = self.myresnet(imgsResNet50)  # imgsFeatures: torch.Size([B*T, 2048, 15, 23])
        OC, OH, OW = imgsFeatures.shape[-3:]
        device = imgsFeatures.device
        if self.spatial_encoding is True:
            sp = spatialencoding2d(OC, OH, OW, device).detach()
            imgsFeatures += sp
        return imgsFeatures

    # roiAlign接口：在特征图上依靠真实坐标值做映射，找到个体区域特征
    # 输入：多批次的每一帧的特征图，每一帧上个体的真实坐标
    # 输出：多批次的帧序列的个体区域特征
    def buildFeatFunc(self, imgsFeatures, boxes_in):
        B, T, N, n = boxes_in.shape
        # print("images_in:", images_in.shape)
        MaxN = self.config.numMaxN
        OC, OH, OW = imgsFeatures.shape[-3:]

        boxes_in_flat = torch.reshape(boxes_in, (B * T * MaxN, 4))
        # 创建一个与 boxes_in_flat 相同大小的全零张量 boxes_scale，它将用于缩放bounding boxes的坐标。
        boxes_scale = torch.zeros_like(boxes_in_flat)
        # 为 boxes_scale 的每一列分别设置缩放因子，OW 和 OH 是缩放的宽度和高度。这里，bounding boxes的宽度和高度都被缩放为 OW 和 OH。
        boxes_scale[:, 0], boxes_scale[:, 1], boxes_scale[:, 2], boxes_scale[:, 3] = OW, OH, OW, OH
        # 将 boxes_in_flat 中的bounding boxes坐标与 boxes_scale 相乘，实现缩放操作。
        boxes_in_flat *= boxes_scale
        # 将 boxes_in_flat 重新形状为三维张量 boxes，其形状为 (B, T * MaxN, 4)。它将bounding boxes恢复为原始的三维表示，其中 B 表示批次大小，T 表示时间步数，MaxN 表示最大框数。
        boxes = boxes_in_flat.reshape(B, T * MaxN, 4)
        boxes.requires_grad = False
        # 创建了一个张量 boxes_idx，用于表示bounding boxes的索引。B * T 表示总的bounding boxes数，MaxN 表示每个时间步的最大框数。这行代码将创建一个大小为 (B * T, MaxN) 的张量，其中每一行是一个时间步内的bounding boxes的索引。
        boxes_idx = torch.arange(B * T, device=boxes_in_flat.device, dtype=torch.int).reshape(-1, 1).repeat(1,
                                                                                                            MaxN)  # B*T, N
        # 将 boxes_idx 重新形状为一维张量 boxes_idx_flat，以便后续的操作。
        boxes_idx_flat = torch.reshape(boxes_idx, (B * T * MaxN,))

        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        # 使用了一个名为 roi_align 的方法，它接受输入图像特征 imgsFeatures、bounding boxes的坐标 boxes_in_flat 和索引 boxes_idx_flat，用于执行一种ROI（感兴趣区域）对齐操作，将bounding boxes对应的特征提取出来。boxes_features 的形状为 (B * T * MaxN, D, K, K)，其中 D 表示特征通道数，K 表示特征的空间维度。
        boxes_features = self.roi_align(imgsFeatures,
                                        boxes_in_flat,
                                        boxes_idx_flat)  # B*T*N,D,K,K
        # boxes_features = self.ln1(boxes_features)
        # boxes_features = self.ecaBlock(boxes_features)
        boxes_features = boxes_features.reshape(B * T * MaxN, -1)
        # print("boxes_features:", boxes_features.shape)
        # 将 boxes_features 传递给一个名为 bbox_fc 的全连接层，然后将结果重新形状为 (B, T, MaxN, -1)。这个步骤可能用于进一步处理bounding boxes的特征。
        boxes_features = self.bbox_fc(boxes_features).reshape(B, T, MaxN, -1)
        # boxes_features = self.relu(boxes_features)
        # boxes_features = self.ln2(boxes_features)

        return boxes_features

    def fill_with_zeros(self, input_tensor):
        # 获取输入张量的大小
        original_size = input_tensor.size()

        # 获取填充 0 后的目标大小
        target_size = (original_size[1], original_size[1])

        # 创建一个大小为目标大小的全零张量
        filled_tensor = torch.zeros(target_size, dtype=input_tensor.dtype, device=input_tensor.device)
        filled_tensor = torch.full(target_size, -1, dtype=input_tensor.dtype, device=input_tensor.device)
        # 将输入张量复制到填充后的张量中
        filled_tensor[:original_size[0], :original_size[1]] = input_tensor

        return filled_tensor

    # 添加群组信息(func：addGroupInfo)
    # 输入：多批次的帧序列的个体区域特征boxesFeatures(B, MaxN, T, F)，分组信息groupInfo(B,MaxN)
    # 输出：群组行为信息groupFeaturesList(B, groupNum, KeyT,F) groupNum：分组数量,KeyT：关键时序信息的帧数
    # 此时得到的groupNum相当于用个体直接推理人群中的个体数量的维度，
    # 即boxesFeatures(B, MaxN, T, F)和groupFeaturesList(B, groupNum, KeyT,F)中，
    # groupNum与MaxN的作用相当，可以进一步提取群组之间的关系
    def reGNI(self, groupInfo, groupLabels):
        B, MaxN = groupInfo.shape
        groupInfoLists = []
        glb = []
        for b in range(B):
            # 去掉0
            non_zero_tensor = groupInfo[b, :][groupInfo[b, :].nonzero()].squeeze()
            non_zero_tensor = groupInfo[b, :][groupInfo[b, :] != -1]
            # print("non_zero_tensor", non_zero_tensor)
            # 统计重复个数
            unique_values, inverse_indices = torch.unique(non_zero_tensor, return_inverse=True)
            # print("unique_values", unique_values, "inverse_indices", inverse_indices)
            # 循环组编号
            groupInfoList = []
            gl = []
            for index in unique_values:
                tmp = torch.full((MaxN,), -1)
                tmplabel = torch.full((MaxN,), -1)
                # 匹配组编号，如果组编号相同，则将元素存储到tmp中，将
                for j in range(MaxN):
                    if index == groupInfo[b, j]:
                        tmp[j] = int(index)
                        tmplabel[j] = groupLabels[b, j]
                gl.append(tmplabel)
                groupInfoList.append(tmp)
                #循环组编号

            gl = torch.stack(gl)
            glb.append(gl)
            # print("groupInfoList", groupInfoList)
            # print("glb", glb)
            groupInfoList = torch.stack(groupInfoList)
            # 填充0
            groupInfoList = self.fill_with_zeros(groupInfoList)

            groupInfoLists.append(groupInfoList)
        groupInfoLists = torch.stack(groupInfoLists)
        glb = torch.cat(glb, dim=0)
        # print("glb", glb.shape)
        return groupInfoLists, glb  # [b,MaxN,MaxN]

    def selectKeyTF(self, boxesFeatures):
        B, T, MaxN, F = boxesFeatures.shape
        boxesFeaturesAll = []
        for b in range(B):
            # n = bboxes_num_in[b][0]
            # TNF
            batchImgsFeatures = boxesFeatures[b, :, :, :]
            bImgFLN = batchImgsFeatures
            # NTF
            bImgFLN = bImgFLN.permute(1, 0, 2).contiguous()

            # 选择关键帧begin
            attn_output, attn_output_weights = self.selfAttnCon(bImgFLN, bImgFLN, value=bImgFLN)
            topk_values, topk_indices = torch.topk(attn_output_weights.sum(dim=1), self.KEYT, dim=1)
            sorted_topk_indices, indices = torch.sort(topk_indices, dim=1)

            # 重组时间序列第一个人的KEYT帧到第n人的KEYT帧
            sorted_topk_indices_ex = sorted_topk_indices.unsqueeze(2).expand(-1, -1, bImgFLN.size(2))
            batchImgsFeaturesNew = torch.gather(bImgFLN, 1, sorted_topk_indices_ex)
            # sorted_topk_indices_ex1 = sorted_topk_indices.unsqueeze(2).expand(-1, -1, boxesFeatures.size(2))
            # spatialFeaturesT = torch.gather(bImgFLN, 1, sorted_topk_indices_ex1)
            N, TKey, F = batchImgsFeaturesNew.shape
            # spatialFeatures = spatialFeaturesT.permute(1, 0, 2).contiguous().reshape(TKey, N, -1)
            boxesFeaturesAll.append(batchImgsFeaturesNew)
        boxesFeaturesAll = torch.stack(boxesFeaturesAll)
        return boxesFeaturesAll

    def reGFI(self, boxesFeaturesKeyT, groupNumInfo):
        B, MaxN, T, F = boxesFeaturesKeyT.shape
        BFGLS = []  # 初始化一个空列表BFGLS，用于存储处理后的结果。
        for b in range(B):  # 对每个批次中的样本进行循环。
            groupLabel = []
            # 循环组数
            BFGL = []  # 初始化一个空列表BFGL，用于存储每个组的结果。
            for i in range(groupNumInfo[b, :, :].size(0)):  # 对多个组进行循环。
                BFG = []
                for j in range(groupNumInfo[b, :, :].size(1)):  # 对组内多个个体进行循环。
                    if groupNumInfo[b, i, j].item() != -1:  # 检查组信息中的特定元素是否不等于0。
                        BFG.append(boxesFeaturesKeyT[b, j, :, :])  # 如果不等于0，将对应位置的盒子特征添加到BFG中。
                if len(BFG) == 0:  # 如果BFG为空，说明该组没有特征，跳出当前组的循环。
                    break
                BFG = torch.stack(BFG)
                # print("BFG:", BFG.shape)
                padding_length = MaxN - BFG.size(0)  # 计算需要填充的长度，以保证每个组内的盒子数量达到MaxN。
                # 在后面填充0
                BFG = torch.cat([BFG, torch.zeros(padding_length, BFG.size(1), BFG.size(2)).cuda()], dim=0)
                # print("BFG:", BFG.shape)
                BFGL.append(BFG)

            BFGL = torch.stack(BFGL)
            #  # 在后面填充0
            padding_length = MaxN - BFGL.size(0)
            BFGL = torch.cat([BFGL, torch.zeros(padding_length, BFGL.size(1), BFGL.size(2), BFGL.size(3)).cuda()],
                             dim=0)
            BFGLS.append(BFGL)
        BFGLS = torch.stack(BFGLS)
        return BFGLS

    # def buildGroupFeatFunc(self, boxesFeatures):
    def buildGroupFeatFunc(self, boxesFeatures, groupId, groups):
        # B, T, MaxN, F = boxesFeatures.shape
        B, T, MaxN = groupId.shape
        groupId = torch.mean(groupId.float(), dim=1)
        groups = torch.mean(groups.float(), dim=1)
        # print("groupId", groupId)
        # print("groups", groups)
        # 自定义数值
        # 2.重新组织groupInfo
        # 使得groupInfo(B,MaxN) --> groupNumInfo(B,MaxN,MaxN,KeyT)和groupNum
        # （1）去重后，查询有多少个组，去掉补充组号0，得到groupNum分组数量
        # （2）将同一个组中的个体按照groupNum的数量重新组织数据，得到groupNumInfo(B,groupNum,MaxN)，后续用作群组行为的标签
        # （3）再将(B,groupNum,MaxN)在帧数维度上进行复制KeyT份，得到(B,groupNum,MaxN,KeyT)
        groupNumInfo, glb = self.reGNI(groupId, groups)
        # 3.选择关键时序时序信息:
        # 输入：(B, MaxN, T, F)
        # 输出：(B, MaxN, KeyT, F)
        boxesFeaturesKeyT = self.selectKeyTF(boxesFeatures)
        # print("boxesFeaturesKeyT:", boxesFeaturesKeyT)
        # 4.将groupFeaturesInfo根据groupNumInfo进行重组,此时完成人群中的群组分段
        # 输入：(boxesFeaturesKeyT(B, MaxN, KeyT, F),groupNumInfo(B,MaxN,MaxN,KeyT))
        # 输出：groupFeaturesInfo(B, groupNum, MaxN, T, F)
        groupFeat = self.reGFI(boxesFeaturesKeyT, groupNumInfo)
        return groupFeat, groupNumInfo, glb

    def conGroupInFunc(self, groupFeat, groupNumInfo):
        # 0.举例：输入输出维度大小
        # B批处理大小 groupNum群组数量 MaxN最大人数 T帧数 KeyT关键时序信息的帧数 F特征向量维度
        # B, groupNum, MaxN, T, KeyT, F = 8, 3, 13, 10, 4, 512
        B, GN, MaxN, T, F = groupFeat.shape
        # 群组行为识别，以个体间关系和帧间为基准
        # 输入：groupFeaturesInfo(B, groupNum, MaxN, T, F)
        # 输出：groupFeaturesList(B, groupNum, T,F)
        groupFeaturesList = []  # 对组数不足13的视频，不进行补0.用于标签对比，计算损失
        groupFeaturesListZero = []  # 对组数不足13的视频，进行补0.用于后续的提取人群行为识别
        groupNum = []
        for b in range(B):
            groupFeaturesListB = []
            groupNumB = 0
            # print("groupNumInfo[b, :, :]:", groupNumInfo[b, :, :])
            for gn in range(groupFeat[b, :, :, :, :].size(0)):
                temp = []
                # 合成组成成员
                for gnn in range(groupFeat[b, gn, :, :, :].size(0)):
                    if torch.all(groupNumInfo[b, gn, :] == -1).item() == True:
                        break
                    if int(groupNumInfo[b, gn, gnn]) != 0:
                        temp.append(groupFeat[b, gn, gnn, :, :])
                # 如果不为空，则将temp转化为tensor
                if torch.all(groupNumInfo[b, gn, :] == -1).item() == False:
                    groupNumB = groupNumB + 1
                    GFII = torch.stack(temp)
                    NNZ, KeyT, F = GFII.shape
                    tfg = self.temporalEncoder(GFII)  # 时序信息
                    sfgTemp = tfg.reshape(NNZ, KeyT, F)  # T N F -> N T F
                    sfg = self.spatialEncoder(sfgTemp)  # 空间信息
                    groupFeaturesListB.append(sfg.mean(dim=0))  # 把个体信息融合为群组信息

            groupNum.append(groupNumB)
            # print("groupNum:", groupNum)
            groupFeaturesListB = torch.stack(groupFeaturesListB, dim=0)
            # print("groupFeaturesListB:", groupFeaturesListB.shape)

            # 对组数不足13的视频，进行补0.用于后续的提取人群行为识别
            if groupFeaturesListB.size(0) != 13:
                a = torch.zeros((13 - int(groupFeaturesListB.size(0)), int(groupFeaturesListB.size(1)),
                                 int(groupFeaturesListB.size(2)))).cuda()
                groupFeaturesListBZ = torch.cat((groupFeaturesListB, a), dim=0)
            # print("groupFeaturesListBZ:", groupFeaturesListBZ.shape)
            groupFeaturesList.append(groupFeaturesListB)
            groupFeaturesListZero.append(groupFeaturesListBZ)
        groupNum = torch.tensor(groupNum)
        groupFeaturesListZero = torch.stack(groupFeaturesListZero)
        groupFeaturesList = torch.cat(groupFeaturesList, dim=0)
        return groupFeaturesList, groupFeaturesListZero, groupNum

    def conGroupOutFunc(self, groupFeaturesListZ, groupNum):
        # print("groupFeaturesListZ:", groupFeaturesListZ.shape)
        # print("groupNum:", groupNum)
        activies = []
        B, GN, KeyT, F = groupFeaturesListZ.shape
        for b in range(B):
            gn = groupNum[b]
            # print("gn:", gn)
            # print("groupFeaturesListZ[b, int(n), :, :]:", groupFeaturesListZ[b, :, :, :][:int(gn)].shape)
            temporalFeatures = self.temporalGroupEncoder(
                groupFeaturesListZ[b, :, :, :][:int(gn)].permute(1, 0, 2).contiguous().reshape(KeyT, gn, F))
            # print("temporalFeatures:", temporalFeatures.shape)
            # SFT = temporalFeatures.permute(1, 0, 2).contiguous().reshape(KeyT, gn, -1)
            # print("SFT：",SFT.shape)
            # features: N,T+1, F
            SFT = temporalFeatures.mean(dim=0).reshape(gn, F).sum(dim=0)
            activies.append(SFT)
        activies = torch.stack(activies, dim=0)
        activities_scoresg = self.activities_classifier(activies)
        return activities_scoresg

    def garModelCellFunc(self, boxes_features, bboxes_num_in):
        B, T, MaxN, F = boxes_features.shape
        # ===========================GARModel Begin==============================#
        actions_scores = []  # 存储行人行为
        activities_scores = []  # 存储人群行为

        # 循环Batch，把每一个Batch中的人取出放在一起
        for b in range(B):
            n = bboxes_num_in[b][0]
            # TNF
            batchImgsFeatures = boxes_features[b, :, :int(n), :]
            bImgFLN = batchImgsFeatures

            spatialFeaturesAll = self.spatialEncoder(bImgFLN)
            # NTF
            bImgFLN = bImgFLN.permute(1, 0, 2).contiguous()
            spatialFeaturesAll = spatialFeaturesAll.permute(1, 0, 2).contiguous()

            # 选择关键帧begin
            attn_output, attn_output_weights = self.selfAttnCon(bImgFLN, bImgFLN, value=bImgFLN)
            topk_values, topk_indices = torch.topk(attn_output_weights.sum(dim=1), self.KEYT, dim=1)
            sorted_topk_indices, indices = torch.sort(topk_indices, dim=1)

            # 重组时间序列第一个人的KEYT帧到第n人的KEYT帧
            sorted_topk_indices_ex = sorted_topk_indices.unsqueeze(2).expand(-1, -1, bImgFLN.size(2))
            batchImgsFeaturesNew = torch.gather(bImgFLN, 1, sorted_topk_indices_ex)
            sorted_topk_indices_ex1 = sorted_topk_indices.unsqueeze(2).expand(-1, -1, spatialFeaturesAll.size(2))
            spatialFeaturesT = torch.gather(spatialFeaturesAll, 1, sorted_topk_indices_ex1)
            N, TKey, F = batchImgsFeaturesNew.shape
            spatialFeatures = spatialFeaturesT.permute(1, 0, 2).contiguous().reshape(TKey, N, -1)

            # 引入位置编码
            pos_encoding = PositionalEncoding2D(F, T)
            spatialFeaturesT = pos_encoding(spatialFeaturesT)
            # # 引入全局变量
            cls_temporal_tokens = self.tokenCLSInit.repeat_interleave(N, dim=0)
            # print("cls_temporal_tokens:",cls_temporal_tokens.shape)
            temporalMatrix = torch.cat((cls_temporal_tokens, spatialFeaturesT), dim=1)
            temporalMatrixT = temporalMatrix.permute(1, 0, 2).contiguous().reshape(TKey + 1, N, -1)
            # features: N,T+1, F
            temporalFeatureEncoder = self.temporalEncoder(temporalMatrix)
            temporalFeatureEncoderT = temporalFeatureEncoder.permute(1, 0, 2).contiguous().reshape(TKey + 1, N, -1)

            # # ===============空间特征融合时间特征 Begin======================#
            # # 将编码后的空间和时间特征传递给解码器进行解码，以进一步加强空间特征。
            # # 输出维度：spatialFeatureDecoder: torch.Size([1, T, N, 512])
            # spatialFeatureDecoder = self.garDecoder(temporalMatrixT, temporalFeatureEncoderT)
            # # print("spatialFeatureDecoder:", spatialFeatureDecoder.squeeze(0).shape)
            # # ===============空间特征融合时间特征 End======================#
            # #
            # # ===============时间特征融合空间特征 Begin======================#
            # # 将编码后的空间和时间特征重塑并转置，然后将它们传递给解码器进行解码，以进一步加强时间特征。
            # # 输出维度：temporalFeatureDecoder: torch.Size([1, N, T, 512])
            # temporalFeatureDecoder = self.garDecoder(temporalMatrix, temporalFeatureEncoder)
            # # print("temporalFeatureDecoder:", temporalFeatureDecoder.squeeze(0).shape)
            # # ===============时间特征融合空间特征 End====================#
            # feature11 = spatialFeatureDecoder + temporalFeatureDecoder.permute(1, 0, 2).contiguous().reshape(TKey + 1,
            #                                                                                                  N, -1)
            f2 = temporalFeatureEncoderT[0].squeeze(0) + spatialFeatures.mean(dim=0).contiguous().reshape(N, -1)
            # features1 = feature11[0] + spatialFeatures.mean(dim=0).contiguous().reshape(N, -1)
            features = self.dropout(f2)

            # features = features[0]
            # features = features.mean(dim = 0)
            # 分类器(有分类变量) begin6
            # features = features.permute(1, 0, 2).contiguous()
            # features = features.mean(dim=0)
            # print("features:", features.shape)
            b_actions_scores = self.actions_classifier(features)
            # print("actions_scores:",actions_scores.shape)
            # actions_scores = actions_scores.reshape(T, N, -1)
            # actions_scores = actions_scores.mean(dim=0).reshape(N, -1)
            # features = features[0]
            b_activities_scores = self.activities_classifier(features)
            # print("activities_scores:", activities_scores.shape)
            b_activities_scores = b_activities_scores.sum(dim=0)
            # 分类器(有分类变量) end

            # 由于batch的人数不同，按照不同人数相加
            actions_scores.append(b_actions_scores)
            activities_scores.append(b_activities_scores)

        actions_scores = torch.cat(actions_scores, dim=0)  # ALL_N,actn_num
        activities_scores = torch.stack(activities_scores, dim=0)  # B,acty_num
        return actions_scores, activities_scores

    def forward(self, images_in, boxes_in, bboxes_num_in, groupInfo, groupLabels):
        # 个体行为标签 人群行为标签 群组行为标签 ID 分组
        # 1、利用骨干网络Resnet50进行逐视频逐帧的特征提取。 self.backboneFunc(images_in)
        # 输入：(视频B 帧数T 通道数3 宽480 高720)
        # 输出：(视频B*帧数T 通道数2048 宽15 高23)
        # 面临的问题：如果使用for循环进行逐帧提取，需要临时分配空间来存储每一次循环的输出，占用额外的空间。
        # 解决的办法：为了提升空间的利用率，将视频B和帧数T合并为B*T，作为新的批处理维度，同样可以达到逐视频逐帧提取特征的目的。
        imgsFeatures = self.backboneFunc(images_in)

        # 2、逐视频逐帧映射个体信息，在个体的维度上对帧序列重组。self.buildFeatFunc(imgsFeatures, boxes_in)
        # 输入：(视频B*帧数T 通道数2048 宽15 高23),boxes_in(视频B 帧数T 个体数N 4)(个体坐标位置)
        # 输出：(视频B 个体数N 帧数T 特征维度F)
        # 面临的问题：利用真值选择个体区域
        # 解决的办法：利用ROIAlign将个体位置boxes_in作为感兴趣区域进行映射，从而得到维度为个体数N的单独个体信息
        # 面临的问题：在序列中，每个个体的序列长度可能会不一致，即在序列中可能会有个体消失和出现。
        # 解决的办法：目前没有通过代码的角度去解决，如果使用代码计算，占用空间和计算量都极大。
        # 故从数据集中来解决该问题，通过SocialCAD数据集的规律，每十帧内的人数保持不变，故在十帧内无个体消失和出现。
        # 面临的问题：在从个体维度重组信息序列时，保证每一帧的个体是一致的。
        # 解决的办法：在逐帧映射个体时，需要保证帧间的个体一致性，即ID不进行交换。
        # SocialCAD具有ID标签，可以利用标签来保证ID不进行交换。同时，不同帧的多个个体的相对顺序保持不变。
        boxes_features = self.buildFeatFunc(imgsFeatures, boxes_in)

        # 3、对个体进行分组，添加一个维度组数GN，然后在每一个组内根据每个个体的组编号来构建组内的个体。self.buildGroupFeatFunc(boxes_features, groupInfo)
        # 输入：(视频B 个体数N 帧数T 特征维度F),(视频B 个体的组编号)
        # 输出：(视频B 组数GN 个体数N 帧数T 特征维度F)
        # 主要包括两个部分：
        # （1）重新组织groupInfo：self.reGNI(groupInfo)
        #  使得groupInfo(B,MaxN) --> groupNumInfo(B,MaxN,MaxN,KeyT)和groupNum
        #  1>去重后，查询有多少个组，去掉补充组号0，得到groupNum分组数量
        #  2>将同一个组中的个体按照groupNum的数量重新组织数据，得到groupNumInfo(B,groupNum,MaxN)，后续用作群组行为的标签
        #  3>再将(B,groupNum,MaxN)在帧数维度上进行复制KeyT份，得到(B,groupNum,MaxN,T)
        # （2）将groupFeaturesInfo根据groupNumInfo进行重组,此时完成人群中的群组分组self.reGFI(boxesFeaturesKeyT, groupNumInfo)
        #   输入：(boxesFeaturesKeyT(B, N, T, F),groupNumInfo(B,GN,N,T))
        #   输出：groupFeaturesInfo(B, GN, N, T, F)
        # 面临的问题：重组后的个体数据怎么和输入数据的个体数据进行对应。
        # 解决的办法：通过固定的索引值，并保持群组编号的相对位置不变，即将非本组个体的群组编号置为0，根据相对位置不变，将对应个体选择到该组内。
        # 面临的问题：重组后的数据中，每个视频中的组数量长短不一
        # 解决的办法：选取最大值MaxN，即假设每个人都为一组，即最大人数即最大的组数。
        # 面临的问题：不同视频中的组数不同
        # 解决的办法：采用补0的方法
        groupFeat, groupNumInfo, glb = self.buildGroupFeatFunc(boxes_features, groupInfo, groupLabels)

        # 4、聚合群组内的行为信息.self.conGroupInFunc(groupFeat, groupNumInfo)
        # 输入：(视频B 组数GN 个体数N 帧数T 特征维度F)
        # 输出：groupFeatList(sum(GN) T F) groupFeatListZero(视频数B MaxGN T F) groupNum：每个视频中的组数
        # 面临的问题：提取个体的时序信息
        # 解决的办法：Transformer捕捉帧间长距离依赖关系
        # 面临的问题：提取个体之间的关系信息
        # 解决的办法：Transfomer捕获个体之间的关系
        # 面临的问题：聚合群组内的信息
        # 解决的办法：采用池化的方式，将组内个体进行聚合
        # imgsFeatures = self.addPositionEncoding(imgsFeatures)
        groupFeatList, groupFeatListZero, groupNum = self.conGroupInFunc(groupFeat, groupNumInfo)

        groupScores = self.groupClassifier(groupFeatList)
        GN, KeyT, cls = groupScores.shape
        groupScores = torch.mean(groupScores, dim=1).reshape(GN, -1)
        # 5、聚合群组间的行为信息：在动态人群场景下，每一个群组对于整体人群行为识别都有不同程度的贡献度，
        # 即无论是在视频序列中群组是否消失或中途出现，对于整体人群行为识别都是有贡献的，主要是保证在同一序列中
        # 输入：groupFeatListZero(视频数B MaxGN T F) groupNum：每个视频中的组数
        # 输出：activitieScores(人群行为识别分类的得分)
        activitieScores = self.conGroupOutFunc(groupFeatListZero, groupNum)
        # 个体直接推理人群
        actions_scores, activities_scores = self.garModelCellFunc(boxes_features, bboxes_num_in)
        actionsScores = actions_scores
        # 个体直接推理人群+群组推理人群
        activitiesScores = activitieScores + activities_scores
        return actionsScores, groupScores, activitiesScores, glb
