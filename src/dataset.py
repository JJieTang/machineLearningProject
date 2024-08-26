import torch
import numpy as np
from torch.nn import functional as F
import torch.utils
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, ini_t, num_t, sp_rate_l, CHANNEL_N, data) -> None:
        super(Dataset, self).__init__()
        self.x, self.y = self.prepare_data(self, ini_t, num_t, sp_rate_l, CHANNEL_N, data)
        
    
    def prepare_data(self, ini_t, num_t, sp_rate_l, CHANNEL_N, data):
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

        # one hot encoding: class_change
        x = torch.cat([F.one_hot(x[:, 0, ...].to(torch.int64), 36).permute(0, 4, 1, 2, 3),
                    F.one_hot(x[:, 1, ...].to(torch.int64), 18).permute(0, 4, 1, 2, 3),
                    F.one_hot(x[:, 2, ...].to(torch.int64), 36).permute(0, 4, 1, 2, 3),
                    x[:, 3:, ...]], axis=1)
        
        return x, y


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
