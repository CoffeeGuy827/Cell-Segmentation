import random

from utils import *

class BaseExperiment(object) :
    def __init__(self, model_name, num_classes, num_channels, optimizer, criterion,
                 epochs, batch_size, lr, momentum, weight_decay, image_size, train_loader):
        super(BaseExperiment, self).__init__()

        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = get_deivce()
        self.fix_seed(self.device)

        self.model = get_model(model_name, num_classes, num_channels)
        if torch.cuda.device_count() > 1:
            print('Multi GPU activate : ', torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.model.apply(self.weights_init)

        self.optimizer = get_optimizer(optimizer, self.model, lr, momentum, weight_decay)
        self.scheduler = get_scheduler(self.optimizer, epochs, len(train_loader), self.lr)
        self.criterion = get_criterion(criterion).to(self.device)

        self.image_size = image_size
        self.num_channels = num_channels

    def fix_seed(self, device):
        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)

        if device == 'cuda':
            torch.cuda.manual_seed_all(4321)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print("your seed is fixed to '4321'")

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def print_params(self):
        print("\nmodel : {}".format(self.model_name))
        print("main optimizer : {}".format(self.optimizer))
        print("epochs : {}".format(self.epochs))
        print("learning rate : {}".format(self.lr))
        print("loss function : {}".format(self.criterion))
        print("batch size : {}".format(self.batch_size))
        print("image size : ({}, {}, {})".format(self.image_size, self.image_size, self.num_channels))
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of trainable parameter : {}".format(total_params))

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.05)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.05)
            nn.init.constant_(m.bias.data, 0)

    def current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _apply_transform(self, image, target):
        image, target = image.to(self.device), target.to(self.device)
        if self.pretrained_encoder :
            image = image.repeat(1, 3, 1, 1)

        return image, target

    def _calculate_criterion(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)

        return loss

    def forward(self, image, target):
        image, target = self._apply_transform(image, target)
        output = self.model(image)
        loss = self._calculate_criterion(output, target)

        return loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
