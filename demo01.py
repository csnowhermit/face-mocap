import torch
from torch.utils.cpp_extension import load
from decalib.utils import lossfunc
from decalib.utils.config import cfg as deca_cfg

# lossfunc.VGGFace2Loss(pretrained_model=self.cfg.model.fr_model_path)
#

#
# # print(deca_cfg.cfg.model.fr_model_path)
# print(deca_cfg.model.fr_model_path)

print(deca_cfg.loss.id)
idLoss = lossfunc.VGGFace2Loss(pretrained_model=deca_cfg.model.fr_model_path)
print(idLoss)