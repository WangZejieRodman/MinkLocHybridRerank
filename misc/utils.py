# misc/utils.py
import os
import configparser
import time
import numpy as np


class ModelParams:
    def __init__(self, model_params_path):
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.output_dim = params.getint('output_dim', 256)
        self.normalize_embeddings = params.getboolean('normalize_embeddings', False)
        self.feature_size = params.getint('feature_size', 256)
        self.pooling = params.get('pooling', 'GeM')

        # =====================================================
        # 新增: Hybrid 混合双流解析逻辑 (BEV粗流 + Cross精流)
        # =====================================================
        if self.model == 'MinkLocHybrid':
            self.coarse_type = params.get('coarse_type', 'bev')
            assert self.coarse_type in ['bev', 'cross'], f"Unsupported coarse_type: {self.coarse_type}"
            self.slice_feature_dim = params.getint('slice_feature_dim', 32)

            # --- 1. 解析 Cross-Section 参数 (精流始终需要) ---
            if 'wz_range_cross' in params:
                self.wz_range_cross = [float(e) for e in params['wz_range_cross'].split(',')]
            else:
                self.wz_range_cross = [-10.0, -4.0, 10.0, 8.0]

            if 's_range_cross' in params:
                self.s_range_cross = [float(e) for e in params['s_range_cross'].split(',')]
            else:
                self.s_range_cross = [-12.0, 12.0]

            if 'div_n_cross' in params:
                self.div_n_cross = [int(e) for e in params['div_n_cross'].split(',')]
            else:
                self.div_n_cross = [256, 32]

            self.s_thickness_cross = params.getfloat('s_thickness_cross', 0.375)
            self.in_channels_cross = params.getint('in_channels_cross', 64)

            from datasets.cross_section_quantization import CrossSectionQuantizer
            self.quantizer_fine = CrossSectionQuantizer(
                wz_range=self.wz_range_cross,
                div_n=self.div_n_cross,
                s_range=self.s_range_cross,
                s_thickness=self.s_thickness_cross
            )

            # --- 2. 解析 BEV 参数 (根据 coarse_type 决定粗流量化器) ---
            if self.coarse_type == 'bev':
                if 'coords_range_bev' in params:
                    self.coords_range_bev = [float(e) for e in params['coords_range_bev'].split(',')]
                else:
                    self.coords_range_bev = [-10., -10., -4., 10., 10., 8.]

                if 'div_n_bev' in params:
                    self.div_n_bev = [int(e) for e in params['div_n_bev'].split(',')]
                else:
                    self.div_n_bev = [256, 256, 32]

                self.in_channels_bev = params.getint('in_channels_bev', self.div_n_bev[2])

                # 注意：后续在第二步实现数据流时，这里的 BEVQuantizer 会被替换为支持偏航角对齐的新版本
                from datasets.quantization import BEVQuantizer
                self.quantizer_coarse = BEVQuantizer(coords_range=self.coords_range_bev, div_n=self.div_n_bev)

            elif self.coarse_type == 'cross':
                self.in_channels_bev = self.in_channels_cross  # 占位兼容
                self.quantizer_coarse = self.quantizer_fine

        # =====================================================
        # 兼容旧版单一流解析逻辑
        # =====================================================
        else:
            self.coordinates = params.get('coordinates', 'bev')
            if self.coordinates == 'bev':
                if 'coords_range' in params:
                    self.coords_range = [float(e) for e in params['coords_range'].split(',')]
                else:
                    self.coords_range = [-10., -10, -4, 10, 10, 8]

                if 'div_n' in params:
                    self.div_n = [int(e) for e in params['div_n'].split(',')]
                else:
                    self.div_n = [256, 256, 32]

                self.in_channels = params.getint('in_channels', self.div_n[2])

                from datasets.quantization import BEVQuantizer
                self.quantizer = BEVQuantizer(coords_range=self.coords_range, div_n=self.div_n)

            elif self.coordinates == 'cross':
                if 'wz_range' in params:
                    self.wz_range = [float(e) for e in params['wz_range'].split(',')]
                else:
                    self.wz_range = [-10.0, -4.0, 10.0, 8.0]

                if 's_range' in params:
                    self.s_range = [float(e) for e in params['s_range'].split(',')]
                else:
                    self.s_range = [-12.0, 12.0]

                if 'div_n' in params:
                    self.div_n = [int(e) for e in params['div_n'].split(',')]
                else:
                    self.div_n = [256, 32]

                self.s_thickness = params.getfloat('s_thickness', 0.375)
                self.in_channels = params.getint('in_channels', 64)

                from datasets.cross_section_quantization import CrossSectionQuantizer
                self.quantizer = CrossSectionQuantizer(
                    wz_range=self.wz_range,
                    div_n=self.div_n,
                    s_range=self.s_range,
                    s_thickness=self.s_thickness
                )
            else:
                raise NotImplementedError(f'Unsupported coordinates: {self.coordinates}')

    def print(self):
        print('Model parameters:')
        print(f'  model: {self.model}')

        if self.model == 'MinkLocHybrid':
            print(f'  coarse_type: {self.coarse_type}')
            if self.coarse_type == 'bev':
                print(f'  [BEV Coarse] coords_range_bev: {self.coords_range_bev}')
                print(f'  [BEV Coarse] div_n_bev: {self.div_n_bev}')
                print(f'  [BEV Coarse] in_channels_bev: {self.in_channels_bev}')
            print(f'  [Cross Fine] wz_range_cross: {self.wz_range_cross}')
            print(f'  [Cross Fine] s_range_cross: {self.s_range_cross}')
            print(f'  [Cross Fine] div_n_cross: {self.div_n_cross}')
            print(f'  [Cross Fine] s_thickness_cross: {self.s_thickness_cross}')
            print(f'  [Cross Fine] in_channels_cross: {self.in_channels_cross}')
            print(f'  slice_feature_dim: {self.slice_feature_dim}')
        else:
            print(f'  coordinates: {self.coordinates}')
            if self.coordinates == 'bev':
                print(f'  coords_range: {self.coords_range}')
            elif self.coordinates == 'cross':
                print(f'  wz_range: {self.wz_range}')
                print(f'  s_range: {self.s_range}')
                print(f'  s_thickness: {self.s_thickness}')
            print(f'  div_n: {self.div_n}')
            print(f'  in_channels: {self.in_channels}')

        print(f'  feature_size: {self.feature_size}')
        print(f'  output_dim: {self.output_dim}')
        print(f'  pooling: {self.pooling}')
        print(f'  normalize_embeddings: {self.normalize_embeddings}')
        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class TrainingParams:
    def __init__(self, params_path: str, model_params_path: str, debug: bool = False):
        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(
            model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path
        self.debug = debug

        config = configparser.ConfigParser()
        config.read(self.params_path)

        params = config['DEFAULT']
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.save_freq = params.getint('save_freq', 0)
        self.num_workers = params.getint('num_workers', 0)

        self.batch_size = params.getint('batch_size', 64)
        self.batch_split_size = params.getint('batch_split_size', None)

        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1.
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.val_batch_size = params.getint('val_batch_size', self.batch_size_limit)

        self.lr = params.getfloat('lr', 1e-3)
        self.epochs = params.getint('epochs', 20)
        self.optimizer = params.get('optimizer', 'Adam')
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                if 'scheduler_milestones' in params:
                    scheduler_milestones = params.get('scheduler_milestones')
                    self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
                else:
                    self.scheduler_milestones = [self.epochs + 1]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss').lower()
        if 'contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)
        elif self.loss == 'truncatedsmoothap':
            self.positives_per_query = params.getint("positives_per_query", 4)
            self.tau1 = params.getfloat('tau1', 0.01)
            self.margin = params.getfloat('margin', None)

        self.similarity = params.get('similarity', 'euclidean')
        assert self.similarity in ['cosine', 'euclidean']

        self.aug_mode = params.getint('aug_mode', 1)
        self.set_aug_mode = params.getint('set_aug_mode', 1)
        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.test_file = params.get('test_file', None)

        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')