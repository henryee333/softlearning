from copy import deepcopy

from ray import tune
import numpy as np

from sac_envs.envs.dclaw.dclaw3_screw_v2 import LinearLossFn, NegativeLogLossFn
from softlearning.misc.utils import get_git_rev, deep_update


DEFAULT_KEY = "__DEFAULT_KEY__"

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2


GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
        'observation_keys': None,
        'observation_preprocessors_params': {}
    }
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': tune.grid_search([1]),
        'eval_render_kwargs': {},
        'eval_n_episodes': 3, # num of eval rollouts
        'eval_deterministic': False,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'her_iters': tune.grid_search([0]),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    }
}


MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 1000,
    'gym': {
        DEFAULT_KEY: 1000,
        'Point2DEnv': {
            DEFAULT_KEY: 50,
        },
        'Pendulum': {
            DEFAULT_KEY: 200,
        },
        'Pusher2d': {
            DEFAULT_KEY: 100,
        },
        'InvisibleArm': {
            DEFAULT_KEY: 250,
        },
        'DClaw3': {
            DEFAULT_KEY: 250,
        },
        'HardwareDClaw3': {
            DEFAULT_KEY: 250,
        },
    },
}

NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK = {
    DEFAULT_KEY: 200,
    'gym': {
        DEFAULT_KEY: 200,
        'Swimmer': {
            DEFAULT_KEY: int(3e2),
        },
        'Hopper': {
            DEFAULT_KEY: int(1e3),
        },
        'HalfCheetah': {
            DEFAULT_KEY: int(3e3),
        },
        'Walker2d': {
            DEFAULT_KEY: int(3e3),
        },
        'Ant': {
            DEFAULT_KEY: int(3e3),
        },
        'Humanoid': {
            DEFAULT_KEY: int(1e4),
        },
        'Pusher2d': {
            DEFAULT_KEY: int(2e3),
        },
        'HandManipulatePen': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateEgg': {
            DEFAULT_KEY: int(1e4),
        },
        'HandManipulateBlock': {
            DEFAULT_KEY: int(1e4),
        },
        'HandReach': {
            DEFAULT_KEY: int(1e4),
        },
        'Point2DEnv': {
            DEFAULT_KEY: int(200),
        },
        'Reacher': {
            DEFAULT_KEY: int(200),
        },
        'Pendulum': {
            DEFAULT_KEY: 10,
        },
        'DClaw3': {
            DEFAULT_KEY: 200,
        },
        'HardwareDClaw3': {
            DEFAULT_KEY: 100,
        },
    },
    'dm_control': {
        DEFAULT_KEY: 200,
        'ball_in_cup': {
            DEFAULT_KEY: int(2e4),
        },
        'cheetah': {
            DEFAULT_KEY: int(2e4),
        },
        'finger': {
            DEFAULT_KEY: int(2e4),
        },
    },
    'robosuite': {
        DEFAULT_KEY: 200,
    }
}


ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK = {
    'gym': {
        'Swimmer': {  # 2 DoF
        },
        'Hopper': {  # 3 DoF
        },
        'HalfCheetah': {  # 6 DoF
        },
        'Walker2d': {  # 6 DoF
        },
        'Ant': {  # 8 DoF
            'Parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'Humanoid': {  # 17 DoF
            'Parameterizable-v3': {
                'healthy_reward': 0.0,
                'healthy_z_range': (-np.inf, np.inf),
                'exclude_current_positions_from_observation': False,
            }
        },
        'Pusher2d': {  # 3 DoF
            'Default-v0': {
                'eef_to_puck_distance_cost_coeff': tune.grid_search([2.0]),
                'goal_to_puck_distance_cost_coeff': 1.0,
                'ctrl_cost_coeff': 0.0,
                'puck_initial_x_range': (1, 1), #(0, 1),
                'puck_initial_y_range': (-0.5, -0.5), # (-1, -0.5),
                'goal_x_range': (-0.5, -0.5), #(-1, 0),
                'goal_y_range': (-0.5, -0.5), #(-1, 1),
                'num_goals': 0,
                'swap_goal_upon_completion': True,
                'reset_mode': "random_puck",
                'initial_distribution_path': "/mnt/sda/ray_results/gym/Pusher2d/Default-v0/2019-06-16T14-59-35-reset-free_single_goal_save_pool/ExperimentRunner_2_her_iters=0,n_initial_exploration_steps=2000,n_train_repeat=1,evaluation={'domain': 'Pusher2d', 'task': 'Defaul_2019-06-16_14-59-36umz5wb9o/",
            },
            'DefaultReach-v0': {
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'ImageDefault-v0': {
                'image_shape': (32, 32, 3),
                'arm_object_distance_cost_coeff': 0.0,
                'goal_object_distance_cost_coeff': 3.0,
            },
            'ImageReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            },
            'BlindReach-v0': {
                'image_shape': (32, 32, 3),
                'arm_goal_distance_cost_coeff': 1.0,
                'arm_object_distance_cost_coeff': 0.0,
            }
        },
        'Point2DEnv': {
            'Default-v0': {
                'observation_keys': ('observation', 'desired_goal'),
            },
            'Wall-v0': {
                'observation_keys': ('observation', 'desired_goal'),
            },
        },
        'Sawyer': {
            task_name: {
                'has_renderer': False,
                'has_offscreen_renderer': False,
                'use_camera_obs': False,
                'reward_shaping': tune.grid_search([True, False]),
            }
            for task_name in (
                    'Lift',
                    'NutAssembly',
                    'NutAssemblyRound',
                    'NutAssemblySingle',
                    'NutAssemblySquare',
                    'PickPlace',
                    'PickPlaceBread',
                    'PickPlaceCan',
                    'PickPlaceCereal',
                    'PickPlaceMilk',
                    'PickPlaceSingle',
                    'Stack',
            )
        },
        'InvisibleArm': {
        'FreeFloatManipulation': {
            'has_renderer': False,
            'has_offscreen_renderer': True,
            'use_camera_obs': False,
            'camera_name': 'agentview',
            'use_object_obs': True,
            'object_to_eef_reward_weight': 0,
            'object_to_target_reward_weight': 10,
            'orientation_reward_weight': 1.0,
            'control_freq': 10,
            'fixed_arm': True,
            'fixed_claw': False,
            'objects_type': 'screw',
            'observation_keys': (
                'joint_pos',
                'joint_vel',
                'gripper_qpos',
                'gripper_qvel',
                'eef_pos',
                'eef_quat',
                # 'robot-state',
                # 'custom-cube_position',
                # 'custom-cube_quaternion',
                # 'custom-cube_to_eef_pos',
                # 'custom-cube_to_eef_quat',
                # 'custom-cube-visual_position',
                # 'custom-cube-visual_quaternion',
                'screw_position',
                'screw_quaternion',
                'screw_to_eef_pos',
                'screw_to_eef_quat',
                'screw-visual_position',
                'screw-visual_quaternion',
            ),
            'target_x_range': [0, 0],
            'target_y_range': [0, 0],
            'target_z_rotation_range': [np.pi, np.pi],
            'num_goals': tune.grid_search([0]),
            'initial_x_range': (0, 0),
            'initial_y_range': (0, 0),
            'initial_z_rotation_range': (0, 0),
            'num_starts': -1,
            'camera_width': 64,
            'camera_height': 64,
            'render_collision_mesh': True,
            'render_visual_mesh': False,
            },
        },
        'DClaw3': {
            'ScrewV2-v0': {
                'object_target_distance_reward_fn': NegativeLogLossFn(0),
                'pose_difference_cost_coeff': 0,
                'joint_velocity_cost_coeff': 0,
                'joint_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (np.pi, np.pi),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (-np.pi, np.pi),
            },
            'ImageScrewV2-v0': {
                'image_shape': (32, 32, 3),
                'object_target_distance_reward_fn': NegativeLogLossFn(0),
                'pose_difference_cost_coeff': 0,
                'joint_velocity_cost_coeff': 0,
                'joint_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (np.pi, np.pi),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (-np.pi, np.pi),
            }
        },
        'HardwareDClaw3': {
            'ScrewV2-v0': {
                'object_target_distance_reward_fn': NegativeLogLossFn(0),
                'pose_difference_cost_coeff': 0,
                'joint_velocity_cost_coeff': 0,
                'joint_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (np.pi, np.pi),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (-np.pi, np.pi),
            },
            'ImageScrewV2-v0': {
                'image_shape': (32, 32, 3),
                'object_target_distance_reward_fn': NegativeLogLossFn(0),
                'pose_difference_cost_coeff': 0,
                'joint_velocity_cost_coeff': 0,
                'joint_acceleration_cost_coeff': 0,
                'target_initial_velocity_range': (0, 0),
                'target_initial_position_range': (np.pi, np.pi),
                'object_initial_velocity_range': (0, 0),
                'object_initial_position_range': (-np.pi, np.pi),
            },
        },
        'DClaw': {
            'DClawPoseStatic-v0': {},
            'DClawPoseDynamic-v0': {},
            'DClawTurnFixed-v0': {},
            'DClawTurnRandom-v0': {},
            'DClawTurnRandomDynamics-v0': {},
            'DClawScrewFixed-v0': {},
            'DClawScrewRandom-v0': {},
            'DClawScrewRandomDynamics-v0': {},
        },
    },
    'dm_control': {
        'ball_in_cup': {
            'catch': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'cheetah': {
            'run': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
        'finger': {
            'spin': {
                'pixel_wrapper_kwargs': {
                    'observation_key': 'pixels',
                    'pixels_only': True,
                    'render_kwargs': {
                        'width': 84,
                        'height': 84,
                        'camera_id': 0,
                    },
                },
            },
        },
    },
}


def get_num_epochs(universe, domain, task):
    level_result = NUM_EPOCHS_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_max_path_length(universe, domain, task):
    level_result = MAX_PATH_LENGTH_PER_UNIVERSE_DOMAIN_TASK.copy()
    for level_key in (universe, domain, task):
        if isinstance(level_result, int):
            return level_result

        level_result = level_result.get(level_key) or level_result[DEFAULT_KEY]

    return level_result


def get_initial_exploration_steps(spec):
    config = spec.get('config', spec)
    initial_exploration_steps = 10 * (
        config
        ['sampler_params']
        ['kwargs']
        ['max_path_length']
    )

    return initial_exploration_steps


def get_checkpoint_frequency(spec):
    config = spec.get('config', spec)
    checkpoint_frequency = (
        config
        ['algorithm_params']
        ['kwargs']
        ['n_epochs']
    ) // NUM_CHECKPOINTS

    return checkpoint_frequency


def get_policy_params(universe, domain, task):
    policy_params = GAUSSIAN_POLICY_PARAMS_BASE.copy()
    return policy_params


def get_algorithm_params(universe, domain, task):
    algorithm_params = {
        'kwargs': {
            'n_epochs': get_num_epochs(universe, domain, task),
            'n_initial_exploration_steps': tune.sample_from(
                get_initial_exploration_steps),
        }
    }

    return algorithm_params


def get_environment_params(universe, domain, task):
    environment_params = (
        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
        .get(universe, {}).get(domain, {}).get(task, {}))

    return environment_params


NUM_CHECKPOINTS = 10
SAMPLER_PARAMS_PER_DOMAIN = {
    'DClaw3': {
        'type': 'SimpleSampler',
    },
    'HardwareDClaw3': {
        'type': 'RemoteSampler',
    }
}


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        get_algorithm_params(universe, domain, task),
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
    )
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': get_environment_params(universe, domain, task),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': get_policy_params(universe, domain, task),
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'observation_preprocessors_params': {}
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': int(1e6)
            }
        },
        'sampler_params': deep_update({
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': get_max_path_length(universe, domain, task),
                'min_pool_size': get_max_path_length(universe, domain, task),
                'batch_size': 256,
            }
        }, SAMPLER_PARAMS_PER_DOMAIN.get(domain, {})),
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(get_checkpoint_frequency),
            'checkpoint_replay_pool': False,
        },
    }

    if task == 'InfoScrewV2-v0':
        variant_spec['replay_pool_params']['kwargs']['include_images'] = True
    if task == 'ImageScrewV2-v0' and ENVIRONMENT_PARAMS['DClaw3']['ImageScrewV2-v0']['state_reward']:
        variant_spec['replay_pool_params']['kwargs']['super_observation_space_shape'] = (9+9+2+1+2,)
    if domain == 'HardwareDClaw3':
        variant_spec['sampler_params']['type'] == 'RemoteSampler'
        variant_spec['algorithm_params']['kwargs']['max_train_repeat_per_timestep'] = 1
    return variant_spec


IMAGE_ENVS = (
    ('robosuite', 'InvisibleArm', 'FreeFloatManipulation'),
)


def is_image_env(universe, domain, task, variant_spec):
    return ('image' in task.lower()
            or 'image' in domain.lower()
            or 'pixel_wrapper_kwargs' in (
                variant_spec['environment_params']['training']['kwargs'])
            or (universe, domain, task) in IMAGE_ENVS)


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if is_image_env(domain, task, variant_spec):
        preprocessor_params = {
            'type': 'ConvnetPreprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': 'layer',
                'downsampling_type': 'conv',
            },
        }

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['kwargs'][
            'observation_preprocessors_params'] = {
                'pixels': deepcopy(preprocessor_params)
            }

        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)
                ['policy_params']
                ['kwargs']
                ['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs'][
            'observation_preprocessors_params'] = (
                tune.sample_from(lambda spec: (deepcopy(
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    ['observation_preprocessors_params']
                )))
            )

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
