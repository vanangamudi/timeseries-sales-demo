import logger
log = logger.get_logger(__file__, 'DEBUG')

import copy
import json
import torch
from tqdm import tqdm

class Trainer:
    def __init__(
            self,
            config,
            hpconfig,
            model,
            loss_function,
            optimizer,

            trainloader,
            testloader,
            cuda = False,
            epochs = 1000000,
            every_nepoch = 1000,
            batch_size = 10,
    ):

        self.config = config
        self.hpconfig = hpconfig
        
        self.model         = model         
        self.loss_function = loss_function  
        self.optimizer     = optimizer 
        
        
        self.trainloader = trainloader        
        self.testloader = testloader
        
        self.epoch        = 0
        self.save_count   = 0
        self.epochs       = epochs
        self.every_nepoch = every_nepoch
        
        self.weights_path = '{}/weights.pt'.format(config['hash'])
        self.loss_records     = []
        self.accuracy_records     = []
        
        self.cuda = config['cuda']
        self.cuda = cuda #function args overrides config.cuda
        if self.cuda:
            self.model.cuda()

    def dump_state(self, path=None):
        log.info('dumping trainer state...')
        if path is None:
            path = '{}/trainer_state.json'.format(self.config['hash'])
            
        with open(path, 'w') as f:
            json.dump((self.epoch,
                       self.epochs,
                       self.every_nepoch,
                       self.weights_path,
                       self.cuda,
                       self.loss_records,
                       self.accuracy_records,
                       self.config,
                       self.hpconfig,
                       self.save_count,
                       ),
                      
                      f,
                      ensure_ascii=True,
                      indent=4)

    def load_state(self, path=None):
        log.info('loading trainer state...')
        if path is None:
            path = '{}/trainer_state.json'.format(self.config['hash'])

        try:
            with open(path) as f:
                (self.epoch,
                 self.epochs,
                 self.every_nepoch,
                 self.weights_path,
                 self.cuda,
                 self.loss_records,
                 self.accuracy_records,
                 self.config,
                 self.hpconfig,
                 self.save_count,
                 ) = json.load(f)
        except:
            log.info('fresh start')
            #log.exception('cannot load trainer state...')

        try:
            self.model.load_state_dict(torch.load(self.weights_path))
        except:
            #log.exception('cannot load model...')
            log.info('fresh start')
            
    def write_metric(self, metric, path):
        with open(self.config['metrics_path'][path], 'w') as f:
            for rec in self.loss_records:
                f.write('{}\t{}\n'.format(*rec))

    ####################################################################
    #          step functions for every batch
    ####################################################################
    def validate_step(self, batch):
        input_, target = batch
        if self.cuda:
            input_, target = input_.cuda(), target.cuda()
            
        output  =  self.model(input_)
        loss    =  self.loss_function(output, target)
        #accuracy = (output == target).float().mean()
        
        return loss#, accuracy

    def eval_step(self, batch):
        input_, target = batch
        if self.cuda:
            input_, target = input_.cuda(), target.cuda()
            
        output  =  self.model(input_)
        return list(zip(input_, target, output))

    def train_step(self, batch):
        self.optimizer.zero_grad()

        input_, target = batch

        if self.cuda:
            input_, target = input_.cuda(), target.cuda()
            
        output  =  self.model(input_)
        loss    =  self.loss_function(output, target)
        loss.backward()
        self.optimizer.step()

        return loss

    ####################################################################
    #          epoch level functions for every batch
    ####################################################################
    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        
        for batch in self.trainloader:
            loss= self.train_step(batch)
            
            losses.append(loss)
            
        return torch.stack(losses).mean().item()
    
    def validate_epoch(self, epoch):
        self.model.eval()
        losses     = []
        accuracies = []
        for batch in self.testloader:
            #loss, accuracy = self.validate_step(batch)
            loss = self.validate_step(batch)
            
            losses.append(loss)
            #accuracies.append(accuracy)
            
        return torch.stack(losses).mean().item(), torch.stack(accuracies).mean().item()
    
    def eval_epoch(self, epoch, testloader=None):
        self.model.eval()
        outputs     = []
        if not testloader:
            testloader = self.testloader
        for batch in testloader:
            output = self.eval_step(batch)
            outputs.extend(output)

        return [torch.stack(i).squeeze() for i in  zip(*outputs)]


    ####################################################################
    #          training loop
    ####################################################################
    def do_train(self, epochs=None):
        if self.loss_records:
            loss = self.loss_records[-1][1]
            prev_loss = self.loss_records[-1][1]
        else:
            loss = prev_loss = 1e10

        if epochs:
            epoch_bar = tqdm(range(self.epoch, self.epoch + epochs))
        else:
            epoch_bar = tqdm(range(self.epoch, self.epochs))
            
        for epoch in epoch_bar:
            self.epoch = epoch
            try:
                epoch_bar.set_description(
                    'epoch:{} - loss:{:0.4f} - saves:{}'.format(
                        epoch, loss, self.save_count))
                
                
                
                if epoch and epoch % self.every_nepoch == 0:
                    print('epoch:{} - loss:{:0.4f} - saves:{}'
                          .format(epoch, loss, self.save_count))

                    #loss, accuracy = self.validate_epoch(epoch)
                    loss = self.validate_epoch(epoch)
                    #self.accuracy_records.append((epoch, accuracy))

                    print('test epoch: {}, loss:{}'
                          .format(epoch, loss))
                    
                    self.write_metric(loss, 'loss_path')
                    #self.write_metric(accuracy, 'accuracy_path')

                loss = self.train_epoch(epoch)
                self.loss_records.append((epoch, loss))

                if prev_loss > loss:
                    prev_loss = loss
                    if self.weights_path:
                        torch.save(copy.deepcopy(self.model).cpu().state_dict(), self.weights_path)
                        self.save_count += 1
            except KeyboardInterrupt:
                self.dump_state()
                return True
                
        return True
