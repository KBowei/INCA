import logging
import math

from models.modules.inv_arch import WaveNet, HarrNet
from models.modules.subnet_constructor import subnet

logger = logging.getLogger('base')


####################
# define network
####################

def define(opt):
    opt_net = opt['network']
    subnet_type = opt_net['subnet']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'
    down_num = int(math.log(opt_net['scale'], 2))
    # net = SAINet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['e_blocks'], opt_net['v_blocks'], down_num, opt_net['gmm_components'])
    if opt_net['wave_type'] == 'harr':
        # baseline model, 仅进行Harr小波分解
        net = HarrNet(opt_net['in_nc'], down_num, opt_net['gmm_components'])
    elif opt_net['wave_type'] == 'iwave':
        # 可学习类小波变换网络
        net = WaveNet(opt_net['in_nc'], subnet(subnet_type, init), down_num, opt_net['gmm_components'])
    
    return net
