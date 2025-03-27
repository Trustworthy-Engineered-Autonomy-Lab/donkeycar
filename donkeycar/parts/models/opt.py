class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.isTrain = False
        
        # basic parameters
        self.name = ''
        self.gpu_ids = []
        self.checkpoints_dir = ''
        
        # model parameters
        self.model = 'pix2pix'
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.n_layers_D = 3
        self.norm = 'batch'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = False
        
        # dataset parameters
        self.dataset_mode = 'aligned'
        self.serial_batches = True
        self.num_threads = 0
        self.batch_size = 1
        self.load_size = 286
        self.crop_size = 256
        self.max_dataset_size = float("inf")
        self.preprocess = 'none'
        self.no_flip = True
        self.display_winsize = 256
        
        # additional parameters
        self.epoch = 'latest'
        self.load_iter = 0
        self.verbose = False
        self.suffix = ''
        self.phase = 'test'
        
        # Default device setup
        import torch
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        # training parameters (even though we're not training, some might be needed)
        self.pool_size = 0
        self.gan_mode = 'vanilla'
        self.lambda_L1 = 100.0
        self.lr = 0.0002
        self.beta1 = 0.5
        
        self.initialized = True
