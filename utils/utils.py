import torch
import torch.distributed as dist
import os
from src.model import *
import numpy as np
import wandb
import matplotlib
from metrics import *
import matplotlib.pyplot as plt



# load the CRNN Model from file
def load_model(model_file='./Setup_0'):
    model_para = np.load(model_file+'/model_setting.npy',allow_pickle=True).item()
    model_file = model_file + '/model.pkl'
    ca = NCA(model_para)
    ca.load_state_dict(torch.load(model_file, map_location='cpu'))
    return ca



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def clean_up():
    dist.destroy_process_group()



def log_predictions(self, x, y, epoch, test_table, note='valid_', min_number=3):
    x = x.permute(0, 3, 4, 2, 1).cpu().detach().numpy()
    x_valid = np.concatenate([np.clip(np.argmax(x[..., :36], axis=-1, keepdims=True) / 36.,0.0,1.0) ,
                        np.clip(np.argmax(x[..., 36:54], axis=-1, keepdims=True) / 18.,0.0,1.0) /2.0,
                        np.clip(np.argmax(x[..., 54:90], axis=-1, keepdims=True) / 36.,0.0,1.0),
                        x[..., 90:]],axis=-1)

    target_valid = y.permute(0, 1, 4, 5, 3, 2).cpu().detach().numpy()

    for ea_i in range(np.min([x_valid.shape[0], min_number])):
        sam_i = x_valid[ea_i, ..., :3]
        target_i = target_valid[ea_i, -1, ...]
        target_i[...,:3] = target_i[...,:3]/36.

        cmap = matplotlib.cm.get_cmap("turbo")  # define a colormap

        img_id = str(note) + str(ea_i)

        zmis_ori = self.cal_misori(np.clip(sam_i[..., x_valid.shape[3] - 1, :3], 0.0, 1.0) * np.pi * 2.0,
                            target_i[..., x_valid.shape[3] - 1, :3] * np.pi * 2.0)  # misorientation angle
        zfil = (zmis_ori > 15.0) & (zmis_ori < 75.0)
        zx_img = sam_i[..., x_valid.shape[3] - 1, :3]
        zy_img = target_i[..., x_valid.shape[3] - 1, :3]
        zdif_img = cmap(zmis_ori * zfil / 90.0)[..., :3]

        ymis_ori = self.cal_misori(np.clip(sam_i[..., x_valid.shape[2] // 2, :, :3], 0.0, 1.0) * np.pi * 2.0,
                                target_i[..., x_valid.shape[2] // 2, :,
                                :3] * np.pi * 2.0)  # misorientation angle
        yfil = (ymis_ori > 15.0) & (ymis_ori < 75.0)
        yx_img = sam_i[..., x_valid.shape[2]//2,:, :3]
        yy_img = target_i[..., x_valid.shape[2]//2,:, :3]
        ydif_img = cmap(ymis_ori * yfil / 90.0)[..., :3]

        test_table.add_data(str(epoch), img_id, wandb.Image(zx_img), wandb.Image(zy_img), wandb.Image(zdif_img),
                            str((1.-np.sum(zfil)/(x_valid.shape[1]*x_valid.shape[2]))*100.), wandb.Image(yx_img),
                            wandb.Image(yy_img), wandb.Image(ydif_img), str((1.-np.sum(yfil)/(x_valid.shape[1]*x_valid.shape[3]))*100.))
            

def zoom(img, scale=4):
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img



def plot_image(world_size, x_valid, target_valid, train_loss_step, valid_loss_step, echo_step, path):
    if world_size > 1:
        x_valid = x_valid.permute(0, 3, 4, 2, 1).cpu().detach().numpy()
        x_valid = np.concatenate([np.argmax(x_valid[..., :36], axis=-1, keepdims=True) / 36.,
                                    np.argmax(x_valid[..., 36:54], axis=-1, keepdims=True) / 18.,
                                    np.argmax(x_valid[..., 54:90], axis=-1, keepdims=True) / 36.,
                                    x_valid[..., 90:]], axis=-1)

        target_valid = target_valid.permute(0, 1, 4, 5, 3, 2).cpu().detach().numpy()

        #iterate through the first 3 samples or less
        for ea_i in range(np.min([x_valid.shape[0], 3])):

            sam_i = x_valid[ea_i, ...,  :3]
            target_i = target_valid[ea_i, -1, ...]
            target_i[..., :3] = target_i[..., :3] / 36.

            mis_ori1 = cal_misori(np.clip(sam_i[...,x_valid.shape[3]-1,:3],0.0,1.0) * np.pi *2.0,
                        target_i[...,x_valid.shape[3]-1,:3]  * np.pi*2.0)  # misorientation angle
            filter1 = (mis_ori1 > 15.0) & (mis_ori1 < 75.0)  
            mis_ori2 = cal_misori(np.clip(sam_i[..., x_valid.shape[2]//2,:, :3], 0.0, 1.0) * np.pi*2.0,
                        target_i[..., x_valid.shape[2]//2,:,:3] * np.pi*2.0)  # misorientation angle
            filter2 = (mis_ori2 > 15.0) & (mis_ori2 < 75.0)  

            cmap = matplotlib.cm.get_cmap("turbo")  

            if ea_i == 0:
                show_img = np.hstack((sam_i[...,x_valid.shape[3]-1,:3], target_i[..., x_valid.shape[3]-1, :3],cmap(mis_ori1* filter1 / 90.0)[...,:3]))

            else:
                show_img2 = np.hstack((sam_i[...,x_valid.shape[3]-1,:3], target_i[..., x_valid.shape[3] - 1, :3],
                                        cmap(mis_ori1 * filter1 / 90.0)[..., :3]))
                show_img = np.vstack((show_img, show_img2))
                show_img2 = np.hstack((sam_i[..., x_valid.shape[2]//2,:, :3],
                                        target_i[..., x_valid.shape[2]//2,:, :3],
                                        cmap(mis_ori2 * filter2 / 90.0)[..., :3]))
            
            # plot the images
            plt.subplot(1, 2, 1)
            plt.imshow(zoom(show_img[..., :3]))

            #plot the loss curves
            plt.subplot(1, 2, 2)
            steps = np.array(range(len(train_loss_step))) * echo_step
            plt.plot(steps, train_loss_step, 'b.')
            plt.plot(steps, valid_loss_step, 'r.')
            plt.title('fold_path')

            # save the figure
            plt.savefig(path + '.jpg')
            plt.close()




class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.2):
        """
        Args:
            patience (int): How long to wait after last time validation acc improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation acc improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

