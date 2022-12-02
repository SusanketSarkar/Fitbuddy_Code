from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence


class Inter_network(nn.Module):
    '''Interaction networks'''
    def __init__(self,rec_rel_mat,
                 send_rel_mat,
                 device,
                 obj_dim,
                 rel_sz,
                 relation_hidden_dim,
                 object_hidden_dim):
        super(Inter_network, self).__init__()
    
        #initalizing parameters
        self.device = device
        self.rec_rel_mat = rec_rel_mat
        self.send_rel_mat = send_rel_mat

        self.obj_dim = obj_dim
        self.num_obj = rec_rel_mat.shape[0]
        self.num_rel = rec_rel_mat.shape[1]
        self.rel_sz = rel_sz # relation dimension
        
        self.hidden_size_r = relation_hidden_dim
        self.hidden_size_o = object_hidden_dim
        self.effect = 50

    
        # relation
        if self.rel_sz == 1:
            self.ra = torch.rand(self.num_rel, self.rel_sz).to(self.device)
            self.x = torch.rand(self.rel_sz, self.num_obj).to(self.device)
        else:
            self.ra = nn.Parameter(torch.zeros(self.num_rel, self.rel_sz))
            self.x = nn.Parameter(torch.zeros(self.rel_sz, self.num_obj))
        

        self.relational = nn.Sequential(
            nn.Linear(self.obj_dim * 2 + self.rel_sz, self.hidden_size_r),
            nn.ReLU(),
            nn.Linear(self.hidden_size_r, self.hidden_size_r),
            nn.ReLU(),
            nn.Linear(self.hidden_size_r, self.hidden_size_r),
            nn.ReLU(),
            nn.Linear(self.hidden_size_r, self.effect)
        )
        
        self.object = nn.Sequential(
            nn.Linear(self.obj_dim + self.effect + self.rel_sz, self.hidden_size_o),
            nn.ReLU(),
            nn.Linear(self.hidden_size_o, self.obj_dim)
        )
    
    def forward(self, object_mat):
        batch_sz = object_mat.shape[0]
        o_send = torch.transpose(torch.bmm(object_mat,self.send_rel_mat.repeat(batch_sz, 1, 1)),1,2)
        o_rec =  torch.transpose( torch.bmm(object_mat,self.rec_rel_mat.repeat(batch_sz, 1, 1)),1,2)
        output = torch.cat((o_send,o_rec,
                            self.ra.repeat(batch_sz, 1, 1) ),dim = 2)
        output = torch.transpose(self.relational(output),1,2)
        output = torch.bmm(output,torch.transpose(self.rec_rel_mat.repeat(batch_sz, 1, 1),1,2))
        output = torch.transpose( output,1,2 )
        object_mat = torch.transpose( object_mat,1,2 )
        output = torch.cat(( output,object_mat,
                             torch.transpose(self.x,0,1).repeat(batch_sz, 1, 1) ),dim = 2)
        return torch.transpose(self.object(output),1,2)

class main_model(nn.Module):
    def __init__(self,rec_rel_mat,send_snd_mat,obj_dim,rel_sz,relation_hidden_dim,
                 object_hidden_dim,device):
        super(main_model, self).__init__()
        self.rel_sz = rel_sz
        self.device = device
        self.Inter_network = Inter_network(rec_rel_mat,
                                           send_snd_mat,
                                           device,
                                           obj_dim,
                                           rel_sz,
                                           relation_hidden_dim,
                                           object_hidden_dim)

    def train(self,source,target,reload,f_nm,n_epochs,batch_size = 100,lr = 0.001):        
        # initalizing Adam optimizer and MSE loss
        optimizer = optim.Adam(self.parameters(), lr = 0.001)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(len(target)/batch_size)

        if reload:
            # if True, reload the model from saved checkpoint and retrain the model from epoch and batch last stoped training 
            optimizer, start_epoch = self.load_ckp(f_nm, optimizer)
            #print("Resuming training from epoch {}".format(start_epoch))
        else:
            # else start training from epoch 0
            start_epoch = 0
        
        train_loss = 0.
        
        for i in tqdm( range(start_epoch,n_epochs)):
          
            #shuffle
            r = torch.randperm(source.shape[0])
            source = source[r[:, None]].squeeze(1)
            target = target[r[:, None]].squeeze(1)

            # initalize batch_loss to 0.
            batch_loss = 0.

            for b in range(n_batches):   
                source_batch = source[b*batch_size: (b*batch_size) + batch_size]
                target_batch = target[b*batch_size: (b*batch_size) + batch_size]
        
                # initialize outputs tensor to save output from the training
                outputs = torch.zeros(batch_size,target_batch.shape[1], target_batch.shape[2], requires_grad=True).to(self.device)
                
                # zero the gradient
                optimizer.zero_grad()

                outputs = self.Inter_network.forward(source_batch)
                
                loss = criterion(outputs,target_batch)
                batch_loss += loss.item()
                
                # backpropagation
                loss.backward()
                optimizer.step()

            # loss for epoch 
            batch_loss = batch_loss/(n_batches)

        # show progress to training
        self.save_ckp(n_epochs,optimizer,f_nm)

    def reload(self,filename = 'checkpoint.pt'):
      optimizer = optim.Adam(self.parameters(), lr = 0.001)
      optimizer, start_epoch = self.load_ckp(filename, optimizer)
      print("Reloaded model trained for {} epochs.".format(start_epoch))

    def save_loss(self,epoch,loss,file_nm = 'logs/loss.csv'):
        '''saves loss value to loss.csv'''
        with open(file_nm, 'a') as file:
            file.write("{},{}\n".format(epoch,loss))
            file.close()


    def save_ckp(self,epoch,optimizer,checkpoint_dir):
        #save checkpoint to resume training
        checkpoint = {
            'epoch': epoch,
            'Inter_network_state_dict': self.Inter_network.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_dir)

    def load_ckp(self,checkpoint_fpath, optimizer):
        # reloads the model from the checkpoint
        checkpoint = torch.load(checkpoint_fpath)
        self.Inter_network.load_state_dict(checkpoint['Inter_network_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return ( optimizer, checkpoint['epoch'])

    def predict(self,source,target= None):
        if source.dim() == 2:
            source = source.unsqueeze(0)
        output =  self.Inter_network.forward(source)

        loss = nn.MSELoss()
        if target != None:
            print('MSE loss over the value {} '.format(loss(output,target)))
        return output.cpu()