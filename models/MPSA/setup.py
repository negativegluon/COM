from .settings.defaults import _C
from .settings.setup_functions import *
root = os.path.dirname(os.path.abspath(__file__))
config = _C.clone()
# cfg_file = os.path.join('configs','baseline', 'swin_tiny.yaml')
# cfg_file = os.path.join('../configs', 'eval', 'eval.yaml')
# cfg_file = os.path.join('configs', 'eval', 'eval_base.yaml')
cfg_file = os.path.join('/root/lbs/LDB/models/MPSA/configs', 'but.yaml')
Raise RuntimeError('在这里更换为./models/MPSA/configs的全局地址')
config = SetupConfig(config, cfg_file)
config.defrost()
## Log Name and Perferences
config.write = True
config.train.checkpoint = True
config.misc.exp_name = f'{config.data.dataset}'
# config.misc.exp_name = f'cars'
# config.misc.log_name = f'pr {config.parameters.parts_ratio}+pd {config.parameters.parts_drop}'
config.misc.log_name = f'Ours'

config.cuda_visible = '6,7'

# Environment Settings
config.data.log_path = os.path.join(config.misc.output, config.misc.exp_name, config.misc.log_name
                                    + time.strftime(' %m-%d_%H-%M', time.localtime()))

config.model.pretrained = os.path.join('/root/lbs/LDB/models/MPSA/pretrained',
                                       config.model.name + config.model.pre_version + config.model.pre_suffix)
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# Setup Functions
config.nprocess, config.local_rank = SetupDevice()
config.data.data_root, config.data.batch_size = LocateDatasets(config)
config.train.lr = ScaleLr(config)
log = SetupLogs(config, config.local_rank)
if config.write and config.local_rank in [-1, 0]:
	with open(config.data.log_path + '/config.json', "w") as f:
		f.write(config.dump())
config.freeze()
SetSeed(config)



