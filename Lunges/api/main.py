import cmath as cm
import math
import os
import pickle
import random

#dtft
from array import *
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#animation
from matplotlib import animation
from scipy.signal import find_peaks, peak_prominences
from torch import optim
from torch.nn.utils.rnn import pack_sequence
from tqdm import tqdm

from .deep_nn import main_model


class main:

     def __init__(self):
          #checking if cuda is avilable
          if torch.cuda.is_available():  
               dev = "cuda:0" 
          else:  
               dev = "cpu"  
          self.device = torch.device(dev)

          self.set_defaults()
          self.setup()

     def setup(self):
          if not os.path.exists(self.path+"data"):
               os.mkdir(self.path+"data")
          
          if not os.path.exists(self.path+"saved_nn_model"):
               os.mkdir(self.path+"saved_nn_model")
          
          if not os.path.exists(self.path+"vids"):
               os.mkdir(self.path+"vids")

          if not os.path.exists(self.path+"test_vids"):
               os.mkdir(self.path+"test_vids")
          


     def set_defaults(self):
          ''' need to override '''


          # path to the data
          self.path = "/content/drive/MyDrive/Fitbuddy_Susanket/csv_with_lib/lunges/api/"

          #default value for rep sepration
          self.rep_seprator_landmark = 'y_0'
          self.rep_seprator_sign = -1

          #probability of using a videos in a reps
          self.video_rep_probability = 0.5

          #probability of using a rep for testing
          self.rep_probability = 0.5

          #animation rep limit per video
          self.rep_anime_limit = 10

          # some other values
          self.ad = 1
          self.td = True
          self.diff = True
          self.normal = True
     
     def right_facing(self,data):
          '''
               takes dataframe of a video as input and returns
                    True: if facing right
                    False: if facing left
          '''
          left_facing_con = 0
          right_facing_con = 0
          if sum(data.iloc[:10]['x_31']) < sum(data.iloc[:10]['x_29']):
               # facing right
               score = sum( ( sum(data.iloc[:10]['visibility_31'])/10 , sum(data.iloc[:10]['visibility_29'])/10 ) ) / 2
               right_facing_con += score
          else:
               # facing left
               score = sum( ( sum(data.iloc[:10]['visibility_31'])/10 , sum(data.iloc[:10]['visibility_29'])/10 ) ) / 2
               left_facing_con += score
          
          if sum(data.iloc[:10]['x_32']) < sum(data.iloc[:10]['x_30']):
               # facing right
               score = sum( ( sum(data.iloc[:10]['visibility_32'])/10 , sum(data.iloc[:10]['visibility_30'])/10 ) ) / 2
               right_facing_con += score
          else:
               # facing left
               score = sum( ( sum(data.iloc[:10]['visibility_32'])/10 , sum(data.iloc[:10]['visibility_30'])/10 ) ) / 2
               left_facing_con += score

          return left_facing_con < right_facing_con

     def max_min(self,df,select):
          mn = 10.0
          mx = -10.0

          for i in select:
               mn = min(mn, min(df[i]))
               mx = max(mx, max(df[i]))
          return mx,mn

     def smooth(self,df,select,norm = True,f = 0):
          new_df = pd.DataFrame()
          if f == 0:
               min( 0.03, (41 / len(df)) )
          time = [i for i in range(len(df))]
          
          all = []
          for i in range(33):
               all.append('x_'+str(i))
               all.append('y_'+str(i))
               all.append('z_'+str(i))

          # this is mostly be selected, but all was chossen for better normalisation
          for i in all:
               new_df[i] = sm.nonparametric.lowess(df[i].values, time,frac= f,
                                                  it=3, delta=0.0, is_sorted=True,
                                                  missing='drop', return_sorted=False)
          
          if norm:
               x_max, x_min = self.max_min(new_df,all[0::3])
               y_max, y_min = self.max_min(new_df,all[1::3])
               z_max, z_min = self.max_min(new_df,all[2::3])

               # normalise x
               for i in select[0::3]:
                    new_df[i] = (new_df[i] - x_min) / (x_max - x_min)

               #normalise y
               for i in select[1::3]:
                    new_df[i] = 1 - ( (new_df[i] - y_min) / (y_max - y_min) )

               #normalise z
               for i in select[2::3]:
                    new_df[i] = (new_df[i] - z_min) / (z_max - z_min)
          
          new_df = new_df[select]
          
          return new_df

     def to_matrix(self, arr0,arr1,ad,diff,normal,td):
          lst = []
          i = 0
          if ad > 0:
               if diff and normal:
                    if td:
                         lst.append([0.5,1.1,0.,0.])
                    else:
                         lst.append([0.5,1.1,0.,0.,0.,0.])
               elif diff:
                    if td:
                         lst.append([0.,0.])
                    else:
                         lst.append([0.,0.,0.])
               else:
                    if td:
                         lst.append([0.5,1.1])
                    else:
                         lst.append([0.5,1.1,0.])
          while i < len(arr0):
               d = []
               if normal:
                    if td: #2d
                         d += arr1[i:i+2]
                    else:# 3d
                         d += arr1[i:i+3]
               if diff:
                    if td:
                         d += [arr1[i+0]-arr0[i+0], arr1[i+1]-arr0[i+1]]
                    else:
                         d += [arr1[i+0]-arr0[i+0], arr1[i+1]-arr0[i+1], arr1[i+2]-arr0[i+2]]
               lst.append(d)
               i += 3
          if ad > 1:
               if diff and normal:
                    if td:
                         lst.append([0.0,0.,0.,0.])
                    else:
                         lst.append([0.5,1.,0.,0.,0.,0.])
               elif diff:
                    if td:
                         lst.append([0.,0.])
                    else:
                         lst.append([0.,0.,0.])
               else:
                    if td:
                         lst.append([0.5,1.])
                    else:
                         lst.append([0.5,1.,0.])
          return lst

     def transform(self,df,select):
          source = []
          target = []
          lst = self.smooth(df,select,True)[select].values.tolist()
               
          for i in range(1,len(lst)-1):
                    source.append( self.to_matrix(lst[i-1],lst[i],self.ad,self.diff,self.normal,self.td) )
                    target.append( self.to_matrix(lst[i],lst[i+1],self.ad,self.diff,self.normal,self.td) )

          return ( torch.transpose( torch.FloatTensor(source), 1,2),
                    torch.transpose( torch.FloatTensor(target), 1,2) )

     def select_generator(self,df):
          # need to be override it.
        #   print("It returns dummy select list, please override it.")
        #   select  = ['x_0','y_0','x_11','y_11']
        #   return select,df
        select = ['x_0','y_0','z_0']
        for index in (12,24,26,28,30,32):
               for axis in ('x_','y_','z_'):
                    select.append(axis+str(index))
        return select,df

     def find_before(self,val,index,data):
          '''
          This takes val as target value before which other value should be smaller,
          index as which before which it should look
          and data as list.
          '''
          ans = index
          for i in range(index,-1,-1):
               if data[i] > val:
                    ans = i
               else:
                    break
          return ans

     def find_after(self,val,index,data):
          '''
          This takes val as target value from which other value should be smaller,
          index as which after which it should look
          and data as list.
          '''
          ans = index
          for i in range(index,len(data)):
               if data[i] > val:
                    ans = i
               else:
                    break
          return ans

     def fpeak(self,data,plot = False):
          y = data #np.array( (data) )
          peaks,_ = find_peaks(y)
          prominences = peak_prominences(y, peaks)[0]
          std_dev = max(0.05, prominences.std() )
               
          good_peaks = [[0]]

          for i in range(len(prominences)):
               if prominences[i]  > std_dev:
                    # seprate end and start
                    key = y[peaks[i]] - ( (prominences[i]) * 0.1 )
                    
                    e0 = self.find_before(key,peaks[i],y)
                    good_peaks[-1].append(e0)

                    s1 = self.find_after(key,peaks[i],y)
                    good_peaks.append([s1])

                    # single end start.
                    #good_peaks.append(peaks[i])

          # seprate end and start     
          good_peaks[-1].append(len(y))
  
          if plot:
               contour_heights = y[peaks] -  prominences
                    
               fig,ax=plt.subplots(figsize=(20, 15))
               plt.plot(np.arange(0,len(y)),y,c='g')
               plt.scatter(peaks, y[peaks], c= 'b')

               #plt.scatter(start_end_x, start_end_y, c= 'y')
                    
               plt.vlines(x=peaks, ymin=contour_heights, ymax= y[peaks], color = 'red')
                    
                    #plt.hlines(*results_half[1:],color = 'purple')
                    #plt.hlines(*results_full[1:],color = 'black')
                    
               plt.legend(['real','peaks','prominences','half width', 'full width'])
               plt.show()
               plt.close()

          # single end start.
          #good_peaks.append(len(y))
          return good_peaks

     def trainer_reader(self):
          # data_traing loaded list
          self.train_data = []

          # correct and incorrect reps
          self.correct_rep = []
          self.incorrect_rep = []

          mapping = pd.read_csv(self.path+'mapping.csv')
          self.select_len = 0
          for i in range(len(mapping)):
               file_name = mapping.iloc[i]['file_name']
               label = int( mapping.iloc[i]['is_correct'] )
               df = pd.read_csv(self.path+'csv/'+file_name+'.csv')
               select,df = self.select_generator(df)
               if self.select_len == 0:
                    self.select_len = len(select)
               elif self.select_len != len(select):
                    print("Please check the select generator function, it is generating select of different length for different videos.")
                    exit()
               if label:
                    source,target = self.transform(df,select)
                    self.train_data.append([file_name,source,target])

               if random.random() > self.video_rep_probability:
                    continue
               count = 0
               for t in range(20,len(df),10):
                    peaks = self.fpeak( -self.rep_seprator_sign* df[ self.rep_seprator_landmark ].iloc[:t]   )
                    if len(peaks)-1 > count:
                         if random.random() > self.rep_probability:
                              continue
                         count += 1
                         start,end = peaks[-2]

                         source,target = self.transform(df.iloc[start:end],select)

                         if label:
                              self.correct_rep.append((source,target,[1]))
                         else:
                              self.incorrect_rep.append((source,target,[0]))


     def gen_reader(self,map_file = 'mapping.csv'):
          # data_traing loaded list
          self.gen_data = []
          cols = []
          mapping = pd.read_csv(self.path+map_file)
          self.select_len = 0
          for i in range(len(mapping)):
               file_name = mapping.iloc[i]['file_name']
               labels = []
               if len(cols) == 0:
                    cols = list(mapping.columns.values)
                    cols.remove('index')
                    cols.remove('file_name')
                    cols.remove('is_correct')
               for col in cols:
                    labels.append(int( mapping.iloc[i][col] ))
               
               df = pd.read_csv(self.path+'csv/'+file_name+'.csv')
               
               select,df = self.select_generator(df)
               if self.select_len == 0:
                    self.select_len = len(select)
               elif self.select_len != len(select):
                    print("Please check the select generator function, it is generating select of different length for different videos.")
                    exit()
               count = 0
               for t in range(20,len(df),10):
                    peaks = self.fpeak( -self.rep_seprator_sign* df[ self.rep_seprator_landmark ].iloc[:t]   )
                    if len(peaks)-1 > count:
                         count += 1
                         start,end = peaks[-2]
                         source,target = self.transform(df.iloc[start:end],select)
                         self.gen_data.append((source,target,labels))

               # last one
               start = peaks[-1][0]
               end = len(df)-1
               source,target = self.transform(df.iloc[start:end],select)
               self.gen_data.append((source,target,labels))


     def rel_matrix(self):
          rels = self.read_rels(self.path+"relations.csv")
          send_rel = [[0 for _ in range(len(rels))] for _ in range((self.select_len // 3)+self.ad)]
          rec_rel = [[0 for _ in range(len(rels))] for _ in range((self.select_len // 3)+self.ad)]
          #print(send_rel)
          for rel_id in range(len(rels)):
               send_rel[rels[rel_id][0]][rel_id] = 1
               rec_rel[rels[rel_id][1]][rel_id] = 1
          return torch.FloatTensor(send_rel).to(self.device), torch.FloatTensor(rec_rel).to(self.device)

     def train_it(self,trials):
          log_fn = self.path+"logs.csv"
          try:
               df = pd.read_csv(log_fn)
               model_id = df['model_id'].iloc[-1] + 1
          except:
               f = open(log_fn,'w')
               f.write("model_id,trained_using,trained_on\n")
               f.close()
               model_id = 10000

          for file_name,source,target in self.train_data:
               epochs = 1000 #random.randrange(600, 3000)
               batch_sz = 100 #random.randrange(50,150)
               rel_sz = 20
               rel_hidden_sz = 150 #random.randrange(100,200)
               obj_hidden_sz = 150 #random.randrange(50,150)
               source = source.to(self.device)
               target = target.to(self.device)

               for _ in range(trials):
                    print("model_id: {}, trained_using: {}".format(model_id,file_name))
                    model = main_model(self.rel_rec_mat,
                              self.rel_snd_mat,
                              source.shape[1],
                              rel_sz,
                              rel_hidden_sz,
                              obj_hidden_sz,
                              self.device).to(self.device)
                    model.train(source,target,False,self.path+"saved_nn_model/"+str(model_id)+".pt",epochs,batch_sz)
                    with open(log_fn, 'a') as file:
                         file.write("{},{},{}\n".format(model_id,file_name,datetime.now()))
                         file.close()
                    self.check_it(model,model_id)
                    model_id += 1
                    del model


     def check_it(self,model,model_id):
          self.predict(model,self.correct_rep,str(model_id)+"_correct",animate = True,write = False)
          self.predict(model,self.incorrect_rep,str(model_id)+"_incorrect",animate = True,write = False)

     def read_rels(self,filename):
          file = open(filename,'r')
          data = []
          for line in file.readlines():
               data.append(list(map(int,list(line.strip().split(',')))))
          file.close()
          return data

     def predict(self,model,rep_list,file_name,animate,write,vid_output_folder="vids"):
          def mse(true,pred):
               true = torch.transpose(true,0,1)[1:(self.select_len // 3)+self.ad+1,:2]
               pred = torch.transpose(pred,0,1)[1:(self.select_len // 3)+self.ad+1,:2]
               t = pow((true - pred),2)
               t = t[:,0] + t[:,1]
               t = t.cpu().clone().detach()
               tt = sum(t.tolist())/((self.select_len // 3)*2)
               t = (t/2).tolist()
               for i in range((self.select_len // 3)):
                    mse_list[i].append(t[i])
               mse_list[-1].append(tt)

          t_pre_val = []
          t_rel_val = []
          t_mse_list = [[]for _ in range((self.select_len // 3)+1)]

          for source,target,label in rep_list:
               pre_val = []
               mse_list = [[]for _ in range((self.select_len // 3)+1)]

               source,target = source.to(self.device),target.to(self.device)

               inpt = source[0]
          
               for k in range(len(source)):
                    temp = model.predict(inpt)
                    inpt = temp.to(self.device)
                    mse(target[k],temp[0].to(self.device).clone().detach())
                    pre_val.append(temp[0].tolist())
                    t_rel_val.append(target[k].clone().detach().cpu().tolist())
          
               if write:
                    self.write_it(mse_list,label,"{}data/{}.csv".format(self.path,file_name))

               t_pre_val.extend(pre_val)
               for i in range(len(t_mse_list)):
                    t_mse_list[i].extend(mse_list[i])

          if animate:
               self.animat_it(t_pre_val[:min(400,len(t_pre_val))],
                         t_rel_val[:min(400,len(t_pre_val))],
                         t_mse_list,
                        "{}{}/{}__{}.mp4".format(self.path,vid_output_folder,file_name,str(datetime.now())))
          
     def animat_it(self,pre_val,rel_val,mse_list,file_name):
          """
          Matplotlib Animation Example

          author: Jake Vanderplas
          email: vanderplas@astro.washington.edu
          website: http://jakevdp.github.com
          license: BSD
          Please feel free to use and modify this, but keep the above information. Thanks!
          """

          # First set up the figure, the axis, and the plot element we want to animate
          plt.rcParams.update({'font.size': 9})
          plt.figure(figsize=(9, 5))

          ax1 = plt.subplot2grid((3, 4), (0, 0),rowspan=3,colspan=2)
          ax1.title.set_text("IN")
          ax1.set_xlim([-0.5,1.5])
          ax1.set_ylim([-0.25,1.25])
          ax1.grid()

          ax2 = plt.subplot2grid((3, 4), (0, 2))
          ax2.title.set_text("MSE 1")
          ax2.set_xlim([-1,len(rel_val)])
          ax2.set_ylim([0,0.1])
          ax2.grid()

          ax3 = plt.subplot2grid((3, 4), (0, 3))
          ax3.title.set_text("MSE 2")
          ax3.set_xlim([-1,len(rel_val)])
          ax3.set_ylim([0,0.1])
          ax3.grid()

          ax4 = plt.subplot2grid((3, 4), (1, 2))
          ax4.title.set_text("MSE 3")
          ax4.set_xlim([-1,len(rel_val)])
          ax4.set_ylim([0,0.1])
          ax4.grid()

          ax5 = plt.subplot2grid((3, 4), (1, 3))
          ax5.title.set_text("MSE 4")
          ax5.set_xlim([-1,len(rel_val)])
          ax5.set_ylim([0,0.1])
          ax5.grid()

          ax6 = plt.subplot2grid((3, 4), (2, 2))
          ax6.title.set_text("MSE 5")
          ax6.set_xlim([-1,len(rel_val)])
          ax6.set_ylim([0,0.1])
          ax6.grid()

          ax7 = plt.subplot2grid((3, 4), (2, 3))
          ax7.title.set_text("MSE 6")
          ax7.set_xlim([-1,len(rel_val)])
          ax7.set_ylim([0,0.1])
          ax7.grid()

          anime_rel = self.read_rels(self.path+'anime_rel.csv')
          
          pred = [None for _ in range(len(anime_rel))]
          real = [None for _ in range(len(anime_rel))]
          for i in range(len(anime_rel)):
               pred[i], = ax1.plot([],[], ms=6, c='r')
               real[i], = ax1.plot([],[], ms=6, c='g')

          error = [None for _ in range(6)]
          val = [None for _ in range(6)]
          
          error[0], = ax2.plot([], [],ms=6)
          val[0] = ax2.text(len(rel_val)-10,0.1,"")

          error[1], = ax3.plot([], [],ms=6)
          val[1] = ax3.text(len(rel_val)-10,0.1,"")

          error[2], = ax4.plot([], [],ms=6)
          val[2] = ax4.text(len(rel_val)-10,0.1,"")

          error[3], = ax5.plot([], [],ms=6)
          val[3] = ax5.text(len(rel_val)-10,0.1,"")

          error[4], = ax6.plot([], [],ms=6)
          val[4] = ax6.text(len(rel_val)-10,0.1,"")

          error[5], = ax7.plot([], [],ms=6)
          val[5] = ax7.text(len(rel_val)-10,0.1,"")

          # initialization function: plot the background of each frame
          def init():
               for i in range(len(anime_rel)):
                    pred[i].set_data([],[])
                    real[i].set_data([],[])

               for i in range(min(len(mse_list),6)):
                    error[i].set_data([],[])
                    val[i].set_text('')

               return pred, real, error,val,

          # animation function.  This is called sequentially
          def animate(i):
               dx = 0
               dy = 1
               for j in range(len(anime_rel)):
                    pred[j].set_data( [[pre_val[i][dx][anime_rel[j][0]] , pre_val[i][dx][anime_rel[j][1]]]],
                                      [[pre_val[i][dy][anime_rel[j][0]] , pre_val[i][dy][anime_rel[j][1]]]] )
                    real[j].set_data( [[rel_val[i][dx][anime_rel[j][0]] , rel_val[i][dx][anime_rel[j][1]]]],
                                      [[rel_val[i][dy][anime_rel[j][0]] , rel_val[i][dy][anime_rel[j][1]]]] )

               x_a = [k for k in range(i)]

               for j in range(min(len(mse_list),6)):
                    error[j].set_data(x_a,mse_list[j][:i])
                    val[j].set_text("{:.2f}".format(mse_list[j][i]))


               return pred, real, error,val,

          # call the animator.  blit=True means only re-draw the parts that have changed.
          anim = animation.FuncAnimation(plt.figure(1), animate, init_func=init,
                                        frames=len(pre_val), interval=20, blit=False)

          # save the animation as an mp4.  This requires ffmpeg or mencoder to be
          # installed.  The extra_args ensure that the x264 codec is used, so that
          # the video can be embedded in html5.  You may need to adjust this for
          # your system: for more information, see
          # http://matplotlib.sourceforge.net/api/animation_api.html
          anim.save(file_name, fps=10, extra_args=['-vcodec', 'libx264'])

          plt.close()

     def dtft_helper(self,f):
          # https://gist.github.com/TheRealMentor/018aab68dc4bb55bb8d9a390f657bd1d

          #Defining DTFT function
          def dtft(f,pt):
               output = [0]*n
               for k in range(n):  
                    s = 0
                    p = 0
                    for t in range(len(f)): 
                         s += f[t] * cm.exp(-1j * pt[k] * t)
                    output[k] = s
               return output

          #Calculating the magnitude of DTFT
          def magnitude(inp,n):
               output = [0]*n
               for t in range(0,n):
                    tmp=inp[t]
                    output[t]= math.sqrt(tmp.real**2 + tmp.imag**2)
               return output

          #Calculating the phase 
          def phase(inp,n):
               output = [0]*n
               for t in range(0,n):
                    tmp=inp[t]
                    output[t]= math.atan2(tmp.imag,tmp.real)
               return output

          n = 11
          #Defining the x-limits
          N = 2*((math.pi)/n)
          x = np.arange(-(math.pi),math.pi,N)
          x1 = np.fft.fftshift(x)
          x1 = x1[:n]

          #Using the function that I made
          made_func = dtft(f,x)
          made_func_shift=np.fft.fftshift(made_func)
          made_func_shift_mag = magnitude(made_func_shift,n)
          made_func_shift_phs = phase (made_func_shift,n)

          return x1, made_func_shift_mag, made_func_shift_phs


     def write_it(self,mse_list,label,file_nm):
          f = open(file_nm,'a')
          f.write("\n")
          f.close()

          def write(lst):
               _, mag, phs = self.dtft_helper(lst)
               f = open(file_nm,'a')
               for i in mag:
                    f.write("{},".format(i))
               for i in phs:
                    f.write("{},".format(i))
               f.close()
          
          for mse in mse_list:
               write(mse)

          for i in range(len(label)-1):
               f = open(file_nm,'a')
               f.write("{},".format(label[i]))
               f.close()
          
          f = open(file_nm,'a')
          f.write("{}".format(label[-1]))
          f.close()

     def return_it(self,mse_list):
          output = []

          def write(lst):
               _, mag, phs = self.dtft_helper(lst)
               for i in mag:
                    output.append(i)
               for i in phs:
                   output.append(i)
          
          for mse in mse_list:
               write(mse)

          return output

     def filename_seprator(self,filename):
          index = len(filename) -1
          while filename[index] != '.':
               index -= 1
          return filename[:index], filename[index+1:]

     def load_trained_model(self,trained_models_path):
          self.trained_models = {}
          self.trained_models["in_model"] = main_model(self.rel_rec_mat,
                                             self.rel_snd_mat,
                                             4,
                                             20,
                                             150,
                                             150,
                                             self.device).to(self.device)
          self.trained_models["in_model"].reload(trained_models_path+"in_model.pt")
          
          dir_list = os.listdir(trained_models_path)
          dir_list.remove('in_model.pt')
          for file in dir_list:
               filename,_ = self.filename_seprator(file)
               if filename == '': continue
               self.trained_models[filename] = pickle.load(open(trained_models_path+file, 'rb'))
     
          
     def trainer(self,trials):
          print("reading the data")
          self.trainer_reader()
          self.rel_snd_mat, self.rel_rec_mat = self.rel_matrix()
          print(self.rel_snd_mat.shape)
          print("training the models")
          self.train_it(trials)

     def generator(self,model_id):
          model_file_name = "{}saved_nn_model/{}.pt".format(self.path,model_id)
          self.gen_reader()
          self.rel_snd_mat, self.rel_rec_mat = self.rel_matrix()
          model = main_model(self.rel_rec_mat,
                              self.rel_snd_mat,
                              4,
                              20,
                              150,
                              150,
                              self.device).to(self.device)
          model.reload(model_file_name)
          self.predict(model,self.gen_data,model_id,False,True)
          del model

     def testing_animator(self,model_id):
          model_file_name = "{}saved_nn_model/{}.pt".format(self.path,model_id)
          self.gen_reader("test_mapping.csv")
          self.rel_snd_mat, self.rel_rec_mat = self.rel_matrix()
          model = main_model(self.rel_rec_mat,
                              self.rel_snd_mat,
                              4,
                              20,
                              150,
                              150,
                              self.device).to(self.device)
          model.reload(model_file_name)
          self.predict(model,self.gen_data,model_id,True,False,"test_vids")
          del model

     
     
     def api_in_predict(self,model,source,target):
          def mse(true,pred):
               true = torch.transpose(true,0,1)[1:(self.select_len // 3)+self.ad+1,:2]
               pred = torch.transpose(pred,0,1)[1:(self.select_len // 3)+self.ad+1,:2]
               t = pow((true - pred),2)
               t = t[:,0] + t[:,1]
               t = t.cpu().clone().detach()
               tt = sum(t.tolist())/((self.select_len // 3)*2)
               t = (t/2).tolist()
               for i in range((self.select_len // 3)):
                    mse_list[i].append(t[i])
               mse_list[-1].append(tt)

          mse_list = [[]for _ in range((self.select_len // 3)+1)]
          source,target = source.to(self.device),target.to(self.device)
          inpt = source[0]
          
          for k in range(len(source)):
               temp = model.predict(inpt)
               inpt = temp.to(self.device)
               mse(target[k],temp[0].to(self.device).clone().detach())

          return self.return_it(mse_list)

     def load_label_mapping(self):
          labels_map = {}
          file = open(self.path+'api_label_mapping.csv', 'r')
          for line in file.read().strip().split('\n'):
               line = line.strip().split(',')
               labels_map[int(line[1])] = line[0]
          file.close()
          return labels_map
     
     def api_pipeline(self,input,label_map,num_of_labels):
          predictions = []
          for i in range(num_of_labels):
              output = self.trained_models['scaler_model_'+str(i)].transform(input)
              prediction=self.trained_models['classifier_model_'+str(i)].predict(output)[0]
              if prediction: predictions.append(label_map[i])
          if len(predictions)==0:
              predictions.append('Incorrect')
          if 'Correct' in predictions: predictions = ['Correct']
          
          return predictions


     def api(self,df,num_of_labels):
          select,df = self.select_generator(df)
          self.select_len = len(select)
          source,target = self.transform(df,select)
          
          self.rel_snd_mat, self.rel_rec_mat = self.rel_matrix()
          self.load_trained_model(self.path+"api_models/")
          err_sign = []
          err_sign.append( self.api_in_predict(self.trained_models['in_model'],source,target) )
          label_map = self.load_label_mapping()
          return self.api_pipeline(err_sign,label_map,num_of_labels)