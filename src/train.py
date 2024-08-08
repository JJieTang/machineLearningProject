from src.model import *
import os

import torch
import wandb
import numpy as np
from dataset import Dataset
from utils.utils import *
from utils.metrics import *

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import itertools





# data loader
def data_loader(x,y,rank,world_size, batch_size=20, pin_memory=False,num_workers=0):
    dataset = torch.utils.data.TensorDataset(x, y)
    if world_size > 1:
        sampler = DistributedSampler(dataset,num_replicas=world_size,rank=rank,shuffle=False,drop_last=False)
        dataloader = DataLoader(dataset,batch_size=batch_size,pin_memory=pin_memory,
                                num_workers=num_workers,drop_last=False,shuffle=False,sampler=sampler)
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size)

    return dataloader


def train_helper(x_batch, xt_batch, model, rank, num_t, time_fac):
    l_time_sum = 0.0
    l_ea1, l_ea2, l_ea3, l_s = 0.0, 0.0, 0.0, 0.0
    x_batch, xt_batch = x_batch.to(rank), xt_batch.to(rank)

    for nca_step in range(num_t):
        for time_i in range(time_fac):
            try:
                x_batch = model(x_batch)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

        l_tmp, l_tea1, l_tea2, l_tea3, l_ts = cweigt_loss(x_batch, xt_batch[:, nca_step])
        l_time_sum += l_tmp
        l_ea1 += l_tea1
        l_ea2 += l_tea2
        l_ea3 += l_tea3
        l_s += l_ts

        if nca_step < num_t:
            x_batch[:,91:92,...] = xt_batch[:, nca_step, 4:5].type(torch.FloatTensor)
        if nca_step < num_t-1:
            x_batch[:, 92:93, ...] = xt_batch[:, nca_step+1, 4:5].type(torch.FloatTensor)
    return x_batch, l_time_sum, l_ea1, l_ea2, l_ea3, l_s, nca_step
            



def train(rank, world_size, nca_train_time, nca_train_data, parameterization, fold_path):

    # setup the process groups
    if world_size>1:
        setup(rank,world_size)
    else:
        rank=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    loss_step = []  # store the training loss
    train_loss_step = []
    valid_loss_step = []


    epoch_num = parameterization.get("epoch", 4000)

    CHANNEL_N = int(parameterization.get("in_dim", 16))
    time_fac = int(parameterization.get("time_fac", 1.0))
    sp_rate = parameterization.get("speedup_rate", [5.0])
    path = fold_path + '/model.pkl'
    num_t = int(parameterization.get("tot_t", 8))
    echo_step = int(parameterization.get("echo_step", 20))

    wandb.login(key="22d390a0bcf1cbef03b661be9dfc56af0c6b5990")

    ca = NCA(parameterization)
    ca.initialize_weights()
    retrain = parameterization.get("retrain", False)
    if retrain:
        ca = load_model()

    optimizer = torch.optim.Adam(ca.parameters(), lr=parameterization.get("lr", 0.001),
                                 weight_decay=parameterization.get("l2_reg", 0.0))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameterization.get("step_size", 3000)),
        gamma=parameterization.get("gamma", 1.0),  # default is no learning rate decay
    )


    ini_t = [0]
    # size for validation set
    val_size = max(int((nca_train_data.shape[0] *len(ini_t)) * 3 // 8),1)
    dataset = Dataset(ini_t, num_t, sp_rate, CHANNEL_N, nca_train_data)
    x_train, y_train, x_val, y_val = dataset.x[:-val_size], dataset.y[:-val_size], dataset.x[-val_size:], dataset.y[-val_size:]
    batch_size = int(parameterization.get("batch_size", 100))



    # data loader
    training_set = data_loader(x_train, y_train, rank,world_size,batch_size)
    validation_set = data_loader(x_val, y_val, rank, world_size, batch_size)


    # move model to rank
    ca = ca.to(rank)
    if world_size>1:
        ca = DDP(ca, device_ids=[rank],output_device=rank)
    early_stopper = EarlyStopping(patience=7, verbose=True,delta=0.1)

    for epoch in range(epoch_num):
        # ✨ W&B: Create a Table to store predictions for each test step
        columns = ["epoch", "id", "NCA_z", "CA_z", "diff_z", "Acc_z", "NCA_y", "CA_y", "diff_y", "Acc_y"]
        test_table = wandb.Table(columns=columns)

        if world_size > 1:
            training_set.sampler.set_epoch(epoch)
        acc_train = []
        for j, (x_batch, xt_batch) in enumerate(training_set):

            x_batch, l_time_sum, l_ea1, l_ea2, l_ea3, l_s, nca_step = train_helper(x_batch, xt_batch, ca, rank, num_t, time_fac)
            l_time_sum.backward()
            for p in ca.parameters():
                p.grad /= (p.grad.norm() + 1e-8)
            optimizer.step()
            optimizer.zero_grad()

            if ((epoch % echo_step == 0) or (epoch == (epoch_num-1))) & ((rank==0) | (world_size==1)):
                print(epoch, "training losses: ", l_time_sum.item(), l_ea1, l_ea2, l_ea3, l_s)
                ca.eval()
                acc_train.append(cal_acc(x_batch, xt_batch[:, nca_step]))
                log_predictions(x_batch, xt_batch, epoch, test_table, 'train_')
                ca.train()
                torch.cuda.empty_cache()

        # validation
        with torch.no_grad():
            scheduler.step()

            if ((epoch % echo_step == 0) & ((rank==0) | (world_size==1))):

                # save model
                if world_size > 1:
                    torch.save(ca.module.state_dict(), path)
                else:
                    torch.save(ca.state_dict(), path)

                ca.eval()
                acc_val=[]
                for j, (x_valid, target_valid) in enumerate(validation_set):

                    x_valid, l_valid, lv_ea1, lv_ea2, lv_ea3, lv_s, nca_step = train_helper(x_valid, target_valid, ca, rank, num_t, time_fac)

                    acc_val.append(cal_acc(x_valid, target_valid[:,nca_step]))
                acc_train = np.mean(np.array(acc_train))
                acc_val = np.mean(np.array(acc_val))
                print(epoch, "valid losses: ", l_valid, lv_ea1, lv_ea2, lv_ea3, lv_s)
                print(epoch, "training loss: ", l_time_sum.item(), "valid loss: ", l_valid,
                      "training acc: ", acc_train, "%   valid accuracy: ", acc_val, "%")
                
                # log result
                loss_step.append(
                    str(epoch) + " training loss: " + str(l_time_sum.item()) + "  valid loss: " + str(l_valid) +
                    "  training acc: " + str(acc_train) + " %   valid accuracy: " + str(acc_val) + " %")
                wandb.log({"epoch": epoch, "trainloss": l_time_sum.item(), "validloss":l_valid,
                      "trainacc":acc_train, "validacc":acc_val, "lea1_t":l_ea1,"lea2_t":l_ea2,"lea3_t":l_ea3,
                           "ls_t":l_s,"lea1_v":lv_ea1,"lea2_v":lv_ea2,"lea3_v":lv_ea3,"ls_v":lv_s})
                train_loss_step.append(l_time_sum.item())
                valid_loss_step.append(l_valid)
                log_predictions(x_valid, target_valid, epoch, test_table)
                wandb.log({"test_predictions": test_table})
                with open(fold_path + "/loss_history.txt", "w") as outfile:
                    outfile.write("\n".join(loss_step))

                # early stop
                early_stopper(acc_val)
                if np.isnan(l_valid) or (acc_train>96) or early_stopper.early_stop:
                    print("######### training end ##########")
                    wandb.finish()
                    if world_size > 1:
                        torch.distributed.barrier()
                    break

                ca.train()
                torch.cuda.empty_cache()




if __name__ == '__main__':
    nx = ny = 32  # domain size
    nz = 64
    cell_len = 1e-6  # mesh size
    rec_step = 200  # total time step in the training data
    delta_t = 1.5e-6  # time increment for CA , unit is s
    ea_type = '2D'  # 2D or quasi-3D
    sample_n = 80  # sample number for EA setting
    T_type = 'non-iso'  # type of temperature, iso is isothermal, noniso is nonisothermal
    T_len = 10  # if >1, rotated temperature is included
    T_min = 20  # minimum undercooling for the temperatrure gradient
    T_max = 45  # maximum undercooling for the temperatrure gradient
    T_iso = 20  # if isothermal case, the undercooling


    nca_train_time = np.array(range(rec_step))
    nca_train_data = np.load('dirsoild_1.npy', allow_pickle=True)[:,::2,...,:-1]
    for i in range(2,3):
        nca_train_data2 =np.load('dirsoild_'+str(i)+'.npy', allow_pickle=True)[:,::2,...,:-1]
        nca_train_data = np.concatenate([nca_train_data,nca_train_data2],axis=0)
        del nca_train_data2
    nca_train_data = torch.from_numpy(nca_train_data)
    nca_train_data = nca_train_data.permute([0, 1, 5, 4, 2, 3]).type(torch.FloatTensor)

    print(nca_train_data.shape)


    parameters = {
        "lr": [5e-4],
        "step_size": [1000],
        "gamma": [0.3],
        "hid_lay_num": [3],
        "kernel_size": [3],
        "neu_num": [64],
        "dic_neu_num": [64],
        "dic_lay_num": [0],
        "in_dim": [10],
        "epoch": [4000],
        "echo_step": [200],
        "cs_pena": [1.0],
        "time_fac": [1.0],
        "rand_seed": [3024],
        "speedup_rate": [[1.0]],
        "batch_size": [4],
        "tot_t": [16],
    }

    world_size = 3

    settings = list(itertools.product(*parameters.values()))
    i = 0
    folder_name = str(os.getcwd())
    for setup in settings:
        print("###################################")
        print('setup:  No.', i + 1)
        folder_path = folder_name + "/Setup_" + str(i)
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        setup_properties = parameters
        j = 0
        for key in parameters:
            setup_properties[key] = setup[j]
            j = j + 1
        print(setup_properties)
        print('data stored at: ', folder_path)
        print("###################################")
        setup_path = folder_path + '/model_setting.npy'
        np.save(setup_path, setup_properties)
        mp.spawn(train, args=(world_size, nca_train_time, nca_train_data, setup_properties, folder_path), nprocs=world_size)
        i = i + 1
    print("ending training")