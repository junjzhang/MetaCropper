import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.models as models

from einops import rearrange, repeat, reduce

# B (b) batch dimension
# R (r) ratio dimension
# C (c) color dimension
# H (h) height dimension
# W (w) width dimension


def create_feature_extractor(net_type, name_layer):
    net_list = []
    num_cascade = 0
    num_channel_feature = 0

    if net_type == 'vgg16':
        base_model = models.vgg16().features
        for name, layer in base_model.named_children():
            net_list.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                num_cascade += 1
            if isinstance(layer, nn.Conv2d):
                num_channel_feature = layer.out_channels
            if name == name_layer:
                break
    net = nn.Sequential(*net_list)
    return net, num_channel_feature, num_cascade


class BiForwardHead(nn.Module):
    def __init__(self, num_embedding_dim, num_channel_out, num_channel_feature):
        super().__init__()
        self.num_channel_feature = num_channel_feature

        self.ARS_FTM_head = nn.Linear(
            num_embedding_dim, num_channel_feature**2)
        self.ARS_PWP_head = nn.Linear(num_embedding_dim, num_channel_out)

    def forward(self, x):
        ARS_FTM = rearrange(self.ARS_FTM_head(
            x), 'b (c1 c2) -> b c1 c2', c1=self.num_channel_feature)
        ARS_PWP = rearrange(self.ARS_PWP_head(x), 'b cout -> b cout () ()')
        return ARS_FTM, ARS_PWP


class MetaLearner(nn.Module):
    def __init__(self, num_embedding_dim, num_layers, num_channel_out, num_channel_feature, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(*[
            nn.Linear(num_embedding_dim, num_embedding_dim), nn.GELU(), nn.Dropout(dropout_rate)]*num_layers, BiForwardHead(
            num_embedding_dim, num_channel_out, num_channel_feature))

    def forward(self, x):
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(self, num_channel_in, num_channel_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channel_in, num_channel_out, (3, 3), padding=(1, 1)), nn.GELU())

    def forward(self, x):
        x = repeat(x, 'b c h w -> b c (h f1) (w f2)', f1=2, f2=2)
        return self.net(x)


class Deconv(nn.Module):
    def __init__(self, num_cascade, num_channel_out, num_channel_feature):
        super().__init__()
        self.net = nn.Sequential(DeconvBlock(num_channel_feature, num_channel_out),
                                 * [DeconvBlock(num_channel_out, num_channel_out)]*(num_cascade-1))

    def forward(self, x):
        return self.net(x)


class Mars(nn.Module):
    def __init__(self, net_type, name_layer, dropout_rate=0.2, dim_embedding=512, num_embedding=501, num_channel_out=96, num_meta_learner_hidden_layers=2):
        super().__init__()

        self.ratio_embedding_nodes = nn.Parameter(
            torch.rand(num_embedding, dim_embedding))
        self.embedding_interp_step = (2*math.log(2))/(num_embedding-1)
        # TODO 优化device

        self.feature_extractor, num_channel_feature, num_cascade = create_feature_extractor(
            net_type, name_layer)
        self.meta_learner = MetaLearner(
            dim_embedding, num_meta_learner_hidden_layers, num_channel_out, num_channel_feature, dropout_rate)
        self.deconv_layers = Deconv(
            num_cascade, num_channel_out, num_channel_feature)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, ratio):
        # generate ARS_FTM and ARS_PWP given ratio
        ratio_embedding = self.get_ratio_embedding_batch(ratio)
        ARS_FTM, ARS_PWP = self.meta_learner(ratio_embedding)

        # get the features from image
        x = self.feature_extractor(x)
        h = x.shape[2]
        w = x.shape[3]
        x = self.GAP(x)

        # transform and add
        x += self.ratio_transform(x, ARS_FTM)

        # replicate to h*w*c
        x = repeat(x, 'b c () () -> b c h w', h=h, w=w)

        # deconv
        x = self.deconv_layers(x)

        # predict point-wise
        x = self.pixelwise_predict(x, ARS_PWP)

        return x

    def get_ratio_embedding(self, ratio):
        log_ratio = math.log(ratio)
        idx_low_node = math.floor(
            (log_ratio+math.log(2))/self.embedding_interp_step) - 1
        rate_high = (log_ratio - (idx_low_node+1) *
                     self.embedding_interp_step + math.log(2))/self.embedding_interp_step
        ratio_embedding = self.ratio_embedding_nodes[idx_low_node, :]*(
            1-rate_high)+self.ratio_embedding_nodes[idx_low_node+1, :]*rate_high
        ratio_embedding = rearrange(ratio_embedding, 'n -> () n')
        return ratio_embedding

    def get_ratio_embedding_batch(self, batch_ratios):
        ratio_embedding = torch.cat([self.get_ratio_embedding(ratio)
                                     for ratio in batch_ratios], dim=0)
        return ratio_embedding

    @staticmethod
    def ratio_transform(x, ARS_FTM):
        x = rearrange(x, 'b c () () -> b () c')
        x = torch.bmm(x, ARS_FTM)
        x = rearrange(x, 'b () c -> b c () ()')
        return x

    @staticmethod
    def pixelwise_predict(x, ARS_PWP):
        x = x*ARS_PWP
        x = reduce(x, 'b c h w -> b h w', 'sum')
        return x
