import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from .MyMultiHeadAttention import MyMultiheadAttention
from .Transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


class GARModelCell(nn.Module):
    def __init__(self, cfg, embed_features):
        super(GARModelCell, self).__init__()
        self.cfg = cfg
        self.embed_features = embed_features
        encoderLayer = TransformerEncoderLayer(self.embed_features, self.cfg.headNum, dropout=self.cfg.dropoutNum,
                                               normalize_before=True)
        encoderNorm = nn.LayerNorm(self.embed_features)
        self.garEncoder = TransformerEncoder(encoderLayer, num_layers=self.cfg.encoderLayersNum, norm=encoderNorm)

        # 解码器
        decoderLayer = TransformerDecoderLayer(self.embed_features, self.cfg.headNum, dropout=self.cfg.dropoutNum,
                                               normalize_before=True)
        decoderNorm = nn.LayerNorm(self.embed_features)
        self.garDecoder = TransformerDecoder(decoderLayer, num_layers=self.cfg.decoderLayersNum, norm=decoderNorm)
        self.selfAttn = MyMultiheadAttention(self.embed_features, 8, dropout=0.1)



    def forward(self, bImgFLN):
        spatialFeaturesAll = self.spatialEncoder(bImgFLN)
        # NTF
        bImgFLN = bImgFLN.permute(1, 0, 2).contiguous()
        spatialFeaturesAll = spatialFeaturesAll.permute(1, 0, 2).contiguous()
        # print("batchImgsFeatures:", batchImgsFeatures.shape)
        # 选择关键帧begin
        # bImgFLN = self.ln2(batchImgsFeatures)
        # gf = globalFeat[b,:,:]
        attn_output, attn_output_weights = self.selfAttnCon(bImgFLN, bImgFLN, value=bImgFLN)
        # print("attn_output:", attn_output.shape,"attn_output_weights:",attn_output_weights.shape)
        topk_values, topk_indices = torch.topk(attn_output_weights.sum(dim=1), self.KEYT, dim=1)
        # print("max_values:",topk_values,"max_indices:",topk_indices,topk_indices.shape)
        sorted_topk_indices, indices = torch.sort(topk_indices, dim=1)
        # print("sorted_topk_indices:",sorted_topk_indices,"indicessort:",indices,indices.shape)
        # 重组时间序列第一个人的三帧到第n人的三帧
        # print("temporalMatrixBefore:", temporalMatrixBefore.shape)
        sorted_topk_indices_ex = sorted_topk_indices.unsqueeze(2).expand(-1, -1, bImgFLN.size(2))
        batchImgsFeaturesNew = torch.gather(bImgFLN, 1, sorted_topk_indices_ex)

        sorted_topk_indices_ex1 = sorted_topk_indices.unsqueeze(2).expand(-1, -1, spatialFeaturesAll.size(2))
        spatialFeatures = torch.gather(spatialFeaturesAll, 1, sorted_topk_indices_ex1)

        N, TKey, F = batchImgsFeaturesNew.shape
        # print("batchImgsFeatures:", batchImgsFeatures.shape)
        # 引入位置编码
        pos_encoding = PositionalEncoding2D(F, T)
        batchImgsFeaturesNew = pos_encoding(batchImgsFeaturesNew)
        # # 引入全局变量
        cls_temporal_tokens = self.tokenCLSInit.repeat_interleave(N, dim=0)
        # print("cls_temporal_tokens:",cls_temporal_tokens.shape)
        temporalMatrix = torch.cat((cls_temporal_tokens, batchImgsFeaturesNew), dim=1)
        # features: N,T+1, F
        temporalFeatureEncoder = self.temporalEncoder(temporalMatrix)
        temporalFeatureEncoderT = temporalFeatureEncoder.permute(1, 0, 2).contiguous().reshape(TKey + 1, N, -1)
        features = temporalFeatureEncoderT[0] + spatialFeatures.mean(dim=1).reshape(N, F)


        # # N：人数 T：时间帧数 F：特征数（从C*H*W经过线性变换而来）
        # T, N, F = x.shape
        # # print("x:", x.shape)
        # # ===============提取空间特征 Begin======================#0
        # # 利用个体和个体的关系，加强每个个体的空间表示(N,B*T,-1)，这里可以计算分组情况
        # # 将输入数据重塑为一个空间矩阵，并将其传递给编码器进行编码。编码后的数据被重塑并转置，以便传递给解码器。
        # # 输出维度：spatialMatrix: torch.Size([T, N, 512])
        # # 输出维度：spatialFeatureEncoder: torch.Size([T, N, 512])
        # # 重塑后输出维度：spatialFeatureEncoder: torch.Size([T, N, 512])
        # ## 加分类变量begin
        # spatialMatrix = x
        # # print("spatialMatrix:", spatialMatrix.shape)
        # # spatialFeatureEncoder = self.garEncoder(spatialMatrix)
        # spatialFeatureEncoder = x
        # # print("spatialFeatureEncoder:", spatialFeatureEncoder.shape)
        # ## 加分类变量end
        #
        # # ===============提取时间特征 Begin======================#
        # # 利用帧和帧之间的关系，提取时序特征的目的
        # # 这些行将输入数据转置并重塑为一个时间矩阵，并将其传递给编码器进行编码。
        # # 输出维度：temporalMatrix: torch.Size([N, T, 512])
        # # temporalFeatureEncoder: torch.Size([N, T, 512])
        # ## 加分类变量begin
        # temporalMatrix = x.permute(1, 0, 2).contiguous().reshape(N, T, -1)
        # temporalFeatureEncoder = self.garEncoder(temporalMatrix)
        # # print("temporalFeatureEncoder:", temporalFeatureEncoder.shape)
        # ## 加分类变量end
        # # ===============提取时间特征 End======================#
        #
        # # ===============空间特征融合时间特征 Begin======================#
        # # 将编码后的空间和时间特征传递给解码器进行解码，以进一步加强空间特征。
        # # 输出维度：spatialFeatureDecoder: torch.Size([1, T, N, 512])
        # temporalFeatureEncoderT = temporalFeatureEncoder.permute(1, 0, 2).contiguous().reshape(T, N, -1)
        # spatialFeatureDecoder = self.garDecoder(spatialFeatureEncoder, temporalFeatureEncoderT)
        # # print("spatialFeatureDecoder:", spatialFeatureDecoder.squeeze(0).shape)
        # # ===============空间特征融合时间特征 End======================#
        # #
        # # ===============时间特征融合空间特征 Begin======================#
        # # 将编码后的空间和时间特征重塑并转置，然后将它们传递给解码器进行解码，以进一步加强时间特征。
        # # 输出维度：temporalFeatureDecoder: torch.Size([1, N, T, 512])
        # spatialFeatureEncoderT = spatialFeatureEncoder.permute(1, 0, 2).contiguous().reshape(N, T, -1)
        # temporalFeatureDecoder = self.garDecoder(spatialFeatureEncoderT, temporalFeatureEncoder)
        # # print("temporalFeatureDecoder:", temporalFeatureDecoder.squeeze(0).shape)
        # # ===============时间特征融合空间特征 End====================#
        # feature1 = spatialFeatureDecoder+temporalFeatureDecoder.permute(1, 0, 2).contiguous().reshape(T, N, -1)
        # feature2 = spatialFeatureEncoder+temporalFeatureEncoder.permute(1, 0, 2).contiguous().reshape(T, N, -1)
        # feature = feature1 + feature2
        # return feature1








        # print("feature:", feature.shape)
        # # ================选择关键帧 End=======================#
        # temporalMatrixBefore = spatialFeatureEncoder.permute(1, 0, 2).contiguous().reshape(N, T, -1)
        # attn_output, attn_output_weights = self.selfAttn(temporalMatrixBefore, temporalMatrixBefore, value=temporalMatrixBefore)
        # # print("attn_output_weights:", attn_output_weights.shape)
        # # print("attn_output_weights:", attn_output_weights[0])
        # # print("attn_output_weightsSUM:", attn_output_weights.sum(dim=1))
        #
        # topk_values, topk_indices = torch.topk(attn_output_weights.sum(dim=1), self.KEYT, dim=1)
        # # print("max_values:",topk_values,"max_indices:",topk_indices,topk_indices.shape)
        # sorted_topk_indices, indices = torch.sort(topk_indices, dim=1)
        # # print("sorted_topk_indices:",sorted_topk_indices,"indicessort:",indices,indices.shape)
        # # 重组时间序列第一个人的三帧到第n人的三帧
        # # print("temporalMatrixBefore:", temporalMatrixBefore.shape)
        # sorted_topk_indices_ex = sorted_topk_indices.unsqueeze(2).expand(-1, -1, temporalMatrixBefore.size(2))
        # temporalMatrixNew = torch.gather(temporalMatrixBefore, 1, sorted_topk_indices_ex)
        # # print("temporalMatrixNew:", temporalMatrixNew.shape)
        # spatialFeatureEncoderNew = torch.stack([spatialFeatureEncoder[sorted_topk_indices[i], i] for i in range(N)], dim=1)
        # # spatialFeatureEncoderNew = spatialFeatureEncoder.index_select(0, sorted_topk_indices)
        # # 生成位置编码张量
        # # print("temporalMatrixNew:", temporalMatrixNew.shape)
        # # 将位置编码张量扩展为和 temporalFeatureEncoder 相同的形状 [N, T, 512]
        # pos_enc = self.pos_enc.expand(N, -1, -1)
        # # print("pos_enc:", pos_enc.shape)
        # # 将位置编码和 temporalFeatureEncoder 相加
        # temporalMatrixNewP = temporalMatrixNew + pos_enc
        # # print(temporalMatrixNewP.shape)  # 输出 [N, T, 512]
        # # ================选择关键帧 End=======================#



        # 将编码器的时空特征直接相加（测试所用）
        # spatialFeatureEncoder0 = spatialFeatureEncoder.permute(1, 0, 2).contiguous().reshape(N, T, -1)
        # # feature0 = spatialFeatureEncoder0 + temporalFeatureEncoder
        # featureCLS = temporalFeatureEncoderCLS
        # spatialFeatureEncoderNew = spatialFeatureEncoderNew.permute(1, 0, 2).contiguous().reshape(N, self.KEYT, -1)
        # # print("spatialFeatureEncoderNew:", spatialFeatureEncoderNew.shape)
        # # print("temporalFeatureEncoderCLS:", temporalFeatureEncoderCLS.shape)
        # temporalFeatureEncoderCLS4 = temporalFeatureEncoderCLS[:, 0, :].unsqueeze(1)
        # # print("temporalFeatureEncoderCLS4:", temporalFeatureEncoderCLS4.shape)
        # spatialFeatureEncoderNew = torch.mean(spatialFeatureEncoderNew,dim=1).unsqueeze(1)
        # featureCLSR = temporalFeatureEncoderCLS4 + spatialFeatureEncoderNew
        # featureCLS = torch.mean(featureCLS, dim=1).unsqueeze(1).repeat(1,self.KEYT+1,1)
        # print("featureCLS:", featureCLS.shape)


        # # ===============将时空特征相加 Begin======================#
        # # 将解码后的空间和时间特征重塑为最终的形状。
        # # 输出维度：spatialFeatureDecoder: torch.Size([13, 1, 4, 512])
        # # 输出维度：temporalFeatureDecoder: torch.Size([13, 1, 4, 512])
        # # 输出维度：feature: torch.Size([N,T,512])
        # spatialFeatureDecoder = spatialFeatureDecoder.reshape(T, N, -1).permute(1, 0, 2).contiguous()
        # temporalFeatureDecoder = temporalFeatureDecoder.reshape(N, T, -1)
        # feature = spatialFeatureDecoder + temporalFeatureDecoder
        # # print("feature:", feature.shape)
        # # ===============将时空特征相加 End======================#



        # # N：人数 B：批次大小 T：时间帧数 F：特征数（从C*H*W经过线性变换而来）
        # N, B, T, F = x.shape
        #
        # # 利用个体和个体的关系，加强每个个体的空间表示(N,B*T,-1)，这里可以计算分组情况
        # # 将输入数据重塑为一个空间矩阵，并将其传递给编码器进行编码。编码后的数据被重塑并转置，以便传递给解码器。
        # # 输出维度：spatialMatrix: torch.Size([13, 4, 512])
        # # 输出维度：spatialFeatureEncoder: torch.Size([13, 4, 512])
        # # 重塑后输出维度：spatialFeatureEncoder: torch.Size([13, 4, 512])
        # spatialMatrix = x.reshape(N, B * T, -1)
        # # print("spatialMatrix:", spatialMatrix.shape,spatialMatrix)
        # spatialFeatureEncoder = self.garEncoder(spatialMatrix)
        # # print("spatialFeatureEncoder:", spatialFeatureEncoder.shape)
        # spatialFeatureEncoder = spatialFeatureEncoder.reshape(N, B, T, -1).permute(2, 1, 0, 3).contiguous().reshape(T,
        #                                                                                                             B * N,
        #                                                                                                             -1)
        #
        # # 利用帧和帧之间的关系，提取时序特征的目的
        # # 这些行将输入数据转置并重塑为一个时间矩阵，并将其传递给编码器进行编码。
        # # 输出维度：temporalMatrix: torch.Size([4, 13, 512])
        # # temporalFeatureEncoder: torch.Size([4, 13, 512])
        # temporalMatrix = x.permute(2, 1, 0, 3).contiguous().reshape(T, B * N, -1)
        # # print("temporalMatrix:", temporalMatrix.shape)
        # temporalFeatureEncoder = self.garEncoder(temporalMatrix)
        # # print("temporalFeatureEncoder:", temporalFeatureEncoder.shape, temporalFeatureEncoder)
        #
        # feature0 = spatialFeatureEncoder + temporalFeatureEncoder
        #
        # # 将编码后的空间和时间特征传递给解码器进行解码，以进一步加强空间特征。N,B*T,-1
        # # 输出维度：spatialFeatureDecoder: torch.Size([1, 4, 13, 512])
        # spatialFeatureDecoder = self.garDecoder(spatialFeatureEncoder, temporalFeatureEncoder)
        # # print("spatialFeatureDecoder:", spatialFeatureDecoder.shape)
        #
        # # 将编码后的空间和时间特征重塑并转置，然后将它们传递给解码器进行解码，以进一步加强时间特征。T,B*N,-1
        # # 输出维度：temporalFeatureDecoder: torch.Size([1, 4, 13, 512])
        # spatialFeatureEncoder = spatialFeatureEncoder.reshape(N, B, T, -1).permute(2, 1, 0, 3).contiguous().reshape(T,
        #                                                                                                             B * N,
        #                                                                                                             -1)
        # temporalFeatureEncoder = temporalFeatureEncoder.reshape(N, B, T, -1).permute(2, 1, 0, 3).contiguous().reshape(T,
        #                                                                                                               B * N,
        #                                                                                                               -1)
        # temporalFeatureDecoder = self.garDecoder(spatialFeatureEncoder, temporalFeatureEncoder)
        # # print("temporalFeatureDecoder:", temporalFeatureDecoder.shape)
        #
        # # 将解码后的空间和时间特征重塑为最终的形状。
        # # 输出维度：spatialFeatureDecoder: torch.Size([13, 1, 4, 512])
        # # 输出维度：temporalFeatureDecoder: torch.Size([13, 1, 4, 512])
        # # 输出维度：feature: torch.Size([13, 2, 4, 512])
        # spatialFeatureDecoder = spatialFeatureDecoder.reshape(N, B, T, -1)
        # temporalFeatureDecoder = temporalFeatureDecoder.reshape(T, B, N, -1).permute(2, 1, 0, 3).contiguous().reshape(N, B,
        #                                                                                                               T, -1)
        # feature = spatialFeatureDecoder + temporalFeatureDecoder
        # feature0 = feature0.reshape(T, B, N, -1).permute(2, 1, 0, 3).contiguous().reshape(N, B, T, -1)
        # # print("feature0:", feature0.shape)
        # return feature
