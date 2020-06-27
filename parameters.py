import sys

params = {}


def init():
    # paths
    params['data_dir'] = 'data/'
    params['tb_dir'] = 'tb/'
    params['check_dir'] = 'checkpoints/'

    # train configuration
    params['batch_size'] = 2
    params['batches'] = 150000

    # datset information
    params['dataset'] = 'iPER'
    params['image_size'] = 256
    params['volume_size'] = 64  # height/width of volume, used by dataset to generate masks, must be 64 for our model
    params['data_workers'] = 7  # parallel workers for bodypart-mask generation and transformation estimation

    # augmentation
    params['augment_color'] = True
    params['augment_transform'] = True

    # volume architecture, change these to create a smaller or larger model
    params['before_count'] = 3  # number of 3D residual blocks before warping
    params['after_count'] = 3  # number of 3D residual blocks after warping
    params['residual_channels'] = 64  # number of 3D channels
    params['depth'] = 32  # depth of the volume

    # ablation models
    params['2d_3d_warp'] = False
    params['2d_3d_pose'] = False

    # adam parameters
    params['alpha'] = 2e-4
    params['beta1'] = 0.5
    params['beta2'] = 0.999

    # loss weighting
    params['feature_loss_weight'] = 3.

    # checkpoints and tensorboard output
    params['steps_per_checkpoint'] = 1000
    params['steps_per_validation'] = 1000
    params['steps_per_scalar_summary'] = 20
    params['steps_per_image_summary'] = 200

    # validation configuration
    params['with_valid'] = False  # if True, training is performed on train and valid and tb outputs are on test split
    params['valid_count'] = 256  # number of samples validation is based on

    params['name'] = 'unnamed'  # name will be appended to both the checkpoint directory and the tebsorboard directory
    params['JOB_ID'] = -1


def load_id(job_id):
    if job_id == 1:
        params['dataset'] = 'iPER'
        params['with_valid'] = True
        params['name'] = 'iPER-3d_w-3d_p'
    elif job_id == 2:
        params['dataset'] = 'iPER'
        params['2d_3d_pose'] = True
        params['with_valid'] = True
        params['name'] = 'iPER-3d_w-2d_p'
    elif job_id == 3:
        params['dataset'] = 'iPER'
        params['with_valid'] = True
        params['name'] = 'iPER-2d_w-3d_p'
    elif job_id == 4:
        params['dataset'] = 'iPER'
        params['2d_3d_pose'] = True
        params['2d_3d_warp'] = True
        params['with_valid'] = True
        params['name'] = 'iPER-2d_w-2d_p'

    elif job_id == 5:
        params['dataset'] = 'fashion3d'
        params['with_valid'] = True
        params['name'] = 'fash-3d_w-3d_p'
    elif job_id == 6:
        params['dataset'] = 'fashion3d'
        params['2d_3d_pose'] = True
        params['with_valid'] = True
        params['name'] = 'fash-3d_w-2d_p-fash'
    elif job_id == 7:
        params['dataset'] = 'fashion3d'
        params['2d_3d_warp'] = True
        params['with_valid'] = True
        params['name'] = 'fash-2d_w-3d_p-fash'
    elif job_id == 8:
        params['dataset'] = 'fashion3d'
        params['2d_3d_pose'] = True
        params['2d_3d_warp'] = True
        params['with_valid'] = True
        params['name'] = 'fash-2d_w-2d_p-fash'

    else:
        raise ValueError()


init()

if len(sys.argv) == 2:
    if sys.argv[1] == 'params':
        for p, v in params.items():
            print('{}:\t{}'.format(p, v))
        raise ValueError

par_names = sys.argv[1::2]
par_vals = sys.argv[2::2]

if len(par_names) != len(par_vals):
    raise ValueError('Number of inputs must be even')

for name, val in zip(par_names, par_vals):
    if name not in params:
        if name == '-f':
            continue
        raise ValueError(f'{name} is not a valid parameter')
    if type(params[name]) == bool:
        params[name] = val == 'True'
    else:
        params[name] = type(params[name])(val)

if params['JOB_ID'] != -1:
    load_id(params['JOB_ID'])

params['tb_dir'] += params['name'] + '/'
params['check_dir'] += params['name'] + '/'
