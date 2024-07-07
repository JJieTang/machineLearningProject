import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch.nn import functional as F


class Dataset:

    def prepare(x,y,rank,world_size, batch_size=20, pin_memory=False,num_workers=0):
        dataset = torch.utils.data.TensorDataset(x, y)
        if world_size > 1:
            sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=False,drop_last=False)
            dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=pin_memory,
                                    num_workers=num_workers,drop_last=False,shuffle=False,sampler=sampler)
        else:
            dataloader = DataLoader(dataset,batch_size=batch_size)

        return dataloader



    def red_t(ini_t, num_t, sp_rate_l, CHANNEL_N, val_size, data):
        for k in range(len(sp_rate_l)):
            sp_rate = int(sp_rate_l[k])
            for i in range(len(ini_t)):
                if (i == 0) & (k == 0):
                    x_initial = np.concatenate([data[:, ini_t[i], ...].numpy(),(data[:, ini_t[i]+sp_rate, ...].numpy())[:, -1:, ...] ,
                                                (data[:, ini_t[i], ...].numpy())[:, 0:1, ...] * 0.0 + sp_rate / 10.], axis=1)
                    y = data[:, (ini_t[i]+sp_rate):data.shape[1]:sp_rate].numpy()
                    if y.shape[1] > num_t:
                        y = y[:, :num_t]
                    elif y.shape[1] < num_t:
                        while y.shape[1] < num_t:
                            y = np.concatenate((y, y[:, -1:]), axis=1)
                else:
                    x_initial = np.concatenate([x_initial,
                                                np.concatenate([data[:, ini_t[i], ...].numpy(), (data[:, ini_t[i]+sp_rate, ...].numpy())[:, -1:, ...] ,
                                                                (data[:, ini_t[i], ...].numpy())[:, 0:1, ...] * 0.0 + sp_rate / 10.], axis=1)], axis=0)
                    y_tmp = data[:, (ini_t[i]+sp_rate):data.shape[1]:sp_rate].numpy()
                    if y_tmp.shape[1] > num_t:
                        y_tmp = y_tmp[:, :num_t]
                    elif y_tmp.shape[1] < num_t:
                        while y_tmp.shape[1] < num_t:
                            y_tmp = np.concatenate((y_tmp, y_tmp[:, -1:]), axis=1)
                    y = np.concatenate((y, y_tmp), axis=0)


        seed = np.zeros([x_initial.shape[0], CHANNEL_N, x_initial.shape[2], x_initial.shape[3], x_initial.shape[4]],
                            np.float32)
        seed[:, :7, ...] = x_initial.astype(np.float32)
        x = torch.from_numpy(seed)
        y = torch.from_numpy(y)

        # split train/valid set
        ord = np.array(range(x.shape[0]))
        np.random.shuffle(ord)
        x = x[ord]
        y = y[ord]

        # one hot encoding: class_change
        x = torch.cat([F.one_hot(x[:, 0, ...].to(torch.int64), 36).permute(0, 4, 1, 2, 3),
                    F.one_hot(x[:, 1, ...].to(torch.int64), 18).permute(0, 4, 1, 2, 3),
                    F.one_hot(x[:, 2, ...].to(torch.int64), 36).permute(0, 4, 1, 2, 3),
                    x[:, 3:, ...]], axis=1)

        return x[:-val_size], y[:-val_size], x[-val_size:], y[-val_size:]