from distutils.log import debug
import torch
from models.resnet import resnet18
from utils import init_seed
from datasets import my_DataLoader
from models import resnet18
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import get_cam_mask,show_pictures
class Denormalise(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-12)
        mean_inv = -mean * std_inv
        super(Denormalise, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(Denormalise, self).__call__(tensor.clone())

class Solver():
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.seed = init_seed(cfg.seed)
        self.image_denormalise = Denormalise(self.cfg.mean, self.cfg.std)
    def start(self):
        try:
            mode = getattr(self, self.cfg.mode)
        except AttributeError:
            print('mode error')
        return mode()

    def train_close_model(self):
        #train a base model
        self.build_data()
        self.build_model(mode = 'close')
        self.build_optimizer(mode = 'close')
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for epoch in range(self.cfg.num-iter):
            print("train epoch %d"%(epoch))
            for idx, (train_images, train_labels) in enumerate(self.train_loader):
                if self.norm == False:
                    train_images = self.Normalize(train_images)
                train_images, train_labels = train_images.to(self.cfg.device), train_labels.to(self.cfg.device)
                self.model.train()
                self.optimizer.zero_grad()
                logits, tuples = self.model(train_images.float())
                class_loss = criterion(logits, train_labels.long())
                loss = class_loss  
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()  
            self.test_base_model()
        self.save()

    def train_open_model(self):
        #train a task model
        self.build_data()
        self.build_model(mode = 'open')
        self.build_optimizer(mode = 'open')
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        images, labels = self.maximize()
        self.train_set.data =  images
        self.train_set.label = labels
        self.train_loader_MEADA  = DataLoader(self.train_set, batch_size=self.cfg.batch_size, 
                            shuffle = True, num_workers = self.cfg.num_workers, pin_memory = False)
        #不能把两个合在一起
        for epoch in range(self.cfg.finetune_epoch):
            mask_unknow = np.ones([self.cfg.batch_size, 3, 224 ,224]) #128 3 224 224
            mask_unknow = torch.from_numpy(mask_unknow).cuda()
            pth = self.model.state_dict()
            self.extractor.load_state_dict(pth)
            print("finetune epoch %d"%(epoch))

            for (train_images, train_labels),(train_images_MEADA, train_labels_MEADA) in zip(self.train_loader,self.train_loader_MEADA):
                if len(train_images) != self.cfg.batch_size:
                    break
                if self.norm == False:
                    train_images = self.Normalize(train_images)
                train_images, train_labels = train_images.to(self.cfg.device), train_labels.to(self.cfg.device)
                
                with torch.no_grad():
                    mask_single_unknow = get_cam_mask(train_images, self.extractor, t1 = 150) #128 224 224
                    mask_single_unknow = torch.from_numpy(mask_single_unknow).cuda()
                    mask_single_know = torch.from_numpy(mask_single_know).cuda()
                mask_unknow[:,0,:] = mask_single_unknow[:,:]
                mask_unknow[:,1,:] = mask_single_unknow[:,:]
                mask_unknow[:,2,:] = mask_single_unknow[:,:]

                
                self.model.train()
                self.optimizer.zero_grad()
                logits, tuples = self.model(train_images.float())
                class_loss = criterion(logits, train_labels.long())
                loss = class_loss  
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()  
            self.test_base_model()
        self.save()             

    def test_close_model(self):
        print('test base model')
        self.model.eval()
        for i in range(len(self.test_loader_know)):
            print(self.cfg.test_domain[i])
            correct_know = 0
            sum_know = 0
            for idx, (test_images, test_labels) in enumerate(self.test_loader_know[i]):
                test_images, test_labels = test_images.to(self.cfg.device), test_labels.to(self.cfg.device)
                if self.norm == False:
                    test_images = self.Normalize(test_images)
                batchnum = len(test_labels)
                sum_know = sum_know + batchnum
                logits,_ = self.model(test_images.float(), train=False)
                max_score_every_class, cls_pred = logits.max(dim=1)
                for i in range(batchnum):
                    if cls_pred[i] == test_labels[i]:
                        correct_know = correct_know + 1
            acc_know = correct_know / sum_know
            print(acc_know)

    def maximize(self):
        self.model.eval()
        images, labels = [], []

        for i, (images_train, labels_train) in enumerate(self.train_loader):

            # wrap the inputs and labels in Variable
            inputs, targets = images_train.cuda(), labels_train.cuda()

            # forward with the adapted parameters
            inputs_embedding = self.model(inputs)[-1]['Embedding'].detach().clone()
            inputs_embedding.requires_grad_(False)

            inputs_max = inputs.detach().clone()
            inputs_max.requires_grad_(True)
            optimizer = optim.SGD(parameters=[inputs_max], lr=self.cfg.lr_max)

            for ite_max in range(self.cfg.loops_adv):
                tuples = self.network(x=inputs_max)

                # loss
                loss = self.loss_fn(tuples[0], targets) - self.cfg.gamma * self.dist_fn(
                    tuples[-1]['Embedding'], inputs_embedding)

                # init the grad to zeros first
                self.model.zero_grad()
                optimizer.zero_grad()

                # backward your network
                (-loss).backward()

                # optimize the parameters
                optimizer.step()

            inputs_max = inputs_max.detach().clone().cpu()
            for j in range(len(inputs_max)):
                input_max = self.image_denormalise(inputs_max[j])
                input_max = self.image_transform(input_max.clamp(min=0.0, max=1.0))
                images.append(input_max)
                labels.append(labels_train[j].item())

        return np.stack(images), labels


    def debug(self):
        raise NotImplementedError

    def build_data(self):
        self.test_loader_know = []
        self.test_loader_unknow = []
        self.train_loader,self.train_set = my_DataLoader(self.cfg.data_path,self.cfg.train_domain,self.cfg.know_list,
                            self.cfg.batch_size,self.cfg.num_workers,self.cfg.transform)
        for i in len(self.cfg.test_domain):
            self.test_loader_know.append(my_DataLoader(self.cfg.data_path,self.cfg.test_domain[i],self.cfg.know_list,
                            self.cfg.batch_size,self.cfg.num_workers,self.cfg.transform)[0])
            
            self.test_loader_unknow.append(my_DataLoader(self.cfg.data_path,self.cfg.test_domain[i],self.cfg.unknow_list,
                            self.cfg.batch_size,self.cfg.num_workers,self.cfg.transform)[0])
    
    
    def _build_one_model(self, model_name):
        if model_name == 'resnet18':
            model = resnet18(classes = self.cfg.num_classes)
        
        return model.cuda()


    def build_model(self,mode):
        if mode == 'close':
            self.model = self._build_one_model(self.cfg.model_name)
        elif mode == 'open':
            self.load()
        else:
            raise ValueError()




    def _build_one_loss(self, loss_name, args={}):
        raise NotImplementedError


    def build_loss(self):
        raise NotImplementedError

    def _build_sgd(self, *models, learning_rate = 0.1,nesterov=None, weight_decay=None):
        ''' Build SGD Optimizer for model '''
        if nesterov is None:
            nesterov = self.cfg.nesterov
        if weight_decay is None:
            weight_decay = self.cfg.weight_decay
        return optim.SGD([{
            'params': m.parameters()
        } for m in models],
                         lr = learning_rate,
                         momentum=0.9,
                         nesterov=nesterov,
                         weight_decay=weight_decay)

    def build_optimizer(self,mode):
        ''' Build optimizer and lr_scheduler based on `cfg.optimizer` '''
        if mode == 'close':
            if self.cfg.optimizer_close == 'SGD':
                self.optimizer = self._build_sgd(self.model, learning_rate= self.cfg.learning_rate_close,
                            nesterov = self.cfg.nesterov_close, weight_decay = self.cfg.weight_decay_close)

            elif self.cfg.optimizer_close == 'Adam':
                raise NotImplementedError

            else:
                raise ValueError()
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.cfg.num-iter)
        
        elif mode == 'open':
            if self.cfg.optimizer_close == 'SGD':
                self.optimizer = self._build_sgd(self.model, learning_rate= self.cfg.learning_rate_open,
                            nesterov = self.cfg.nesterov_open, weight_decay = self.cfg.weight_decay_open)

            elif self.cfg.optimizer_close == 'Adam':
                raise NotImplementedError

            else:
                raise ValueError()
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.cfg.finetune_epoch)

    
    def print_log(self, str, print_time=True):
        raise NotImplementedError

    def print(self, str):
        raise NotImplementedError
    
    def load(self):
        #加载基模型
        self.model = torch.load(self.cfg.save_path + '/' + self.cfg.model_name + '_' + self.cfg.domain)
        self.extractor = torch.load(self.cfg.save_path + '/' + self.cfg.model_name + '_' + self.cfg.domain)
        #添加未知类检测头
        self.model.clf2=nn.Linear(512,self.cfg.dummynumber)
        self.extractor.clf2=nn.Linear(512,self.cfg.dummynumber)
        self.model = self.model.cuda()
        self.extractor = self.extractor.cuda()


    def load_checkpoint(self, filename: str, optim: bool = True) -> int:
        raise NotImplementedError

    def save(self):
        torch.save(self.model, self.cfg.save_path + '/' + self.cfg.model_name + '_' + self.cfg.domain)

    def save_checkpoint(self, filename):
        raise NotImplementedError

    def auto_resume(self):
        raise NotImplementedError