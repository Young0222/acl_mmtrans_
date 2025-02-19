DATASET_PATH = ''
DATASET_DIR = ''
SCHEMA_PATH = ''

edges = [
    [0, 1], [0, 2], [0, 3], [0, 4],
    [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11],
    [12, 13], [12, 14], [12, 15], [12, 16], [12, 17],
    [18, 19], [18, 20], [18, 21], [18, 22], [19, 5], [20, 12], [21, 0],
    [23, 24], [23, 25],
    [26, 27], [26, 28], [26, 29], [26, 30], [27, 5], [28, 23], [29, 23], [30, 0],
    [31, 32], [31, 33], [31, 34], [31, 35], [31, 36],
    [37, 38], [37, 39], [37, 40], [38, 5], [39, 0], [40, 31],
    [41, 42],
    [43, 44], [43, 45], [43, 46], [44, 5], [45, 41], [46, 0],
    [47, 48],
    [49, 50], [49, 51], [49, 52], [50, 5], [51, 47], [52, 0],
    [53, 54], [53, 55], [53, 56], [53, 57], [53, 58],
    [59, 60], [59, 61], [59, 62], [60, 5], [61, 53], [62, 0],
    [63, 64], [63, 65], [63, 66], [63, 67], [63, 68],
    [69, 70], [69, 71], [69, 72], [70, 5], [71, 63], [72, 0]
]

class Option:
    def __init__(self, **kwargs) -> None:
        self.batch_size = 8
        self.gradient_descent_step = 6
        self.learning_rate = 1e-4
        self.epochs = 1500
        self.validation_steps = 50
        self.seed = 42

        self.sa_mode = '000'
        self.sa_epochs = 30000
        self.sa_report_interval = 500
        self.validation_sa_steps = 1000

        self.eval_model_suffix = ''

        self.model_cls = 'No'
        self.device = '1'
        self.dataset = 'IMDB'
        self.ql = 'AQL'
        self.use_mir = True
        self.use_skeleton = False
        self.method = 'MMTrans'
        self.model_name_or_path = ''
        
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

        mir_sign = '(MIR)' if self.use_mir else ''
        skeleton_sign = 'Sk' if self.use_skeleton else 'NoSk'
        self.exp_name = f'{self.method}-{self.dataset}-{self.ql}{mir_sign}-{self.model_cls}-{skeleton_sign}'

        self.save_path = f'./models/{self.exp_name}'
        self.use_adafactor = True
        self.mode = 'train'
        self.num_beams = 4
        self.num_return_sequences = 4

        self.output_path = f'./output/{self.exp_name}.log'

        self.nfolds = 4

    def __repr__(self):
        d = ''
        for k in dir(self):
            if not k.startswith('_'):
                d += f'{k}: {getattr(self, k)}\n'
        return d
    