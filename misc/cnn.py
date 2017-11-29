import os.path as osp
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models.vgg import make_layers
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

model_configs = {
    'resnet50' : [3, 4, 6, 3],
    'resnet101' : [3, 4, 23, 3],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(1).sqrt()+self.eps
        x /= norm.expand_as(x)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class ResNetModel(models.ResNet):
    """
    Wrapper for ResNet models
    """
    def __init__(self, opt):
        self.opt = opt
        spec = opt.cnn_model
        flag = False
        super(ResNetModel, self).__init__(models.resnet.Bottleneck, model_configs[spec])
        # Initialize the cnn weights:
        if vars(opt).get('cnn_start_from', None) is not None:
            flag = True
        else:
            opt.logger.debug('Setting ResNet weigths from the models zoo')
            self.load_state_dict(model_zoo.load_url(model_urls[spec]))
            #  opt.logger.debug('Setting ResNet weigths from ruotianluo model')
            #  self.load_state_dict(torch.load('/home/thoth/melbayad/scratch/.torch/models/ruotianluo_resnet50.pth'))
        #  self.avgpool = nn.Module()
        self.fc = nn.Module()
        self.norm2 = L2Norm(n_channels=2048, scale=True)
        if flag:
            opt.logger.debug('Setting CNN weigths from %s' % opt.cnn_start_from)
            self.load_state_dict(torch.load(opt.cnn_start_from))

        opt.logger.warn('Finetuning up from %d modules in the cnn' % opt.finetune_cnn_slice)
        self.to_finetune = list(self._modules.values())[opt.finetune_cnn_slice:] #less = 6, otherwise 5

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        att_feats = x
        #  x = x.mean(2).mean(3).squeeze(2).squeeze(2)
        x = self.avgpool(x)
        if self.opt.norm_feat:
            # x = self.norm2(x)
            x = torch.nn.functional.normalize(x, dim=1)
        x = x.view(x.size(0), -1)
        #  x = self.fc(x)
        # TODO: add norm2 to the fc_feat.
        return att_feats, x

    def forward_caps(self, x, seq_per_img, return_unique=False):
        att_feats, fc_feats = self.forward(x)
        att_feats_unique = att_feats
        fc_feats_unique = fc_feats
        # Duplicate for caps per image
        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0),
                                                     seq_per_img,) +
                                                    att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) *
                                                                                                seq_per_img,) + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), seq_per_img,) +
                                                fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) *
                                                                                           seq_per_img,) + fc_feats.size()[1:]))
        if return_unique:
            return att_feats, fc_feats, att_feats_unique, fc_feats_unique
        return att_feats, fc_feats

    def filter_bn(self):
        for k, layer in self._modules.items():
            # print('layer:', k, type(layer))
            if isinstance(layer, nn.Sequential):
                # go deeper
                for kk, ll in layer._modules.items():
                    # print('sub-layer:', kk, type(ll))
                    if isinstance(ll, models.resnet.Bottleneck):
                        # Once more!
                          for kkk, lll in ll._modules.items():
                            # print('bottleneck-layer:', kkk, type(lll))
                            if isinstance(lll, nn.BatchNorm2d):
                                lll.eval()

                    elif isinstance(ll, nn.BatchNorm2d):
                        ll.eval()

            elif isinstance(layer, nn.BatchNorm2d):
                layer.eval()


class VggNetModel(models.VGG):
    """
    Wrapper for VGG models
    """
    def __init__(self, opt):
        spec = opt.cnn_model
        self.opt = opt
        flag = False
        super(VggNetModel, self).__init__(make_layers(model_configs[spec]))
        if vars(opt).get('cnn_start_from', None) is not None:
            flag = True # Reorder layers before loading
            #  self.load_state_dict(torch.load(osp.join(opt.start_from, 'model-cnn.pth')))
        else:
            opt.logger.debug('Setting VGG weigths from the models zoo')
            self.load_state_dict(model_zoo.load_url(model_urls[spec]))
        opt.logger.warn('Setting the fc feature as %s' % opt.cnn_fc_feat)
        if opt.cnn_fc_feat == 'fc7':
            self.keepdim_fc = 6
        elif opt.cnn_fc_feat == 'fc6':
            self.keepdim_fc = 3
        elif opt.cnn_fc_feat == 'fc8':
            self.keepdim_fc = 7

        self.keepdim_att = 30
        #  print 'PRE:', self._modules
        # Reassemble:
        self.features1 = nn.Sequential(*list(self.features._modules.values())[:self.keepdim_att])
        self.features2 = nn.Sequential(*list(self.features._modules.values())[self.keepdim_att:])
        self.fc = nn.Sequential(*list(self.classifier._modules.values())[:self.keepdim_fc])
        self.features = nn.Module()
        self.classifier = nn.Module()
        self.norm2 = L2Norm(n_channels=512, scale=True)
        if flag:
            self.load_state_dict(torch.load(opt.cnn_start_from))
        self.to_finetune = self._modules.values()
        self.keep_asis = []
        #  print 'POST:', self._modules

    def forward(self, x):
        x = self.features1(x)
        #  print 'step1:', x.size()
        att_feats = x
        x = self.features2(x)
        #  print 'step2:', x.size()
        if self.opt.norm_feat:
            x = self.norm2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #  print "step2:", x.size()
        return att_feats, x

    def forward_caps(self, x, seq_per_img):
        att_feats, fc_feats = self.forward(x)
        # Duplicate for caps per image
        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0),
                                                     seq_per_img,) +
                                                    att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) *
                                                                                                seq_per_img,) + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), seq_per_img,) +
                                                fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) *
                                                                                           seq_per_img,) + fc_feats.size()[1:]))
        return att_feats, fc_feats


