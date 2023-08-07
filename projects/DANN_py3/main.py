import random
import os
import torch
import logging 
import numpy as np 
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model import CNNModel
from torchvision import transforms
from test import test
from data_loader import create_dataloader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

experiment_name = "NWPU-RESISC45-AID"
root= os.path.dirname(__file__)
source_dataset_name = 'NWPU-RESISC45'
target_dataset_name = 'AID'
log_dir = f"logs/{experiment_name}/run_1e-3_128"
writer = SummaryWriter(log_dir)

if __name__ == '__main__':

    source_image_root = os.path.join(root, 'dataset', source_dataset_name)
    target_image_root = os.path.join(root, 'dataset', target_dataset_name)
    model_root = os.path.join(root, 'models')
    cudnn.benchmark = True
    learning_rate = 1e-3
    batch_size = 32
    image_size = 224
    n_epoch = 100

    ''' set device '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    random.seed(42) 
    torch.manual_seed(42)
    np.random.seed(42)

    ''' load data '''

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size,),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size,),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_source, val_source, test_source = create_dataloader(dataset=source_image_root, transform=img_transform_source, batch_size=batch_size, train_ratio=0.7, validation_ratio=0.2)

    train_target, val_target, test_target = create_dataloader(dataset=target_image_root, transform=img_transform_target, batch_size=batch_size, train_ratio=0.7, validation_ratio=0.2)

    ''' load model '''

    my_net = CNNModel()

    for p in my_net.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(my_net.parameters(), lr=learning_rate)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    best_accu_t = 0.0
    best_epoch = 0
    best_accu_s = 0.0

    my_net = my_net.to(device)
    loss_class = loss_class.to(device)
    loss_domain = loss_domain.to(device)
    
    correct_class_batch = 0
    total_samples_batch = 0
    correct_class_epoch = 0
    total_samples_epoch = 0

    for epoch in range(n_epoch):
        len_dataloader = min(len(train_source), len(train_target))
        data_source_iter = iter(train_source)
        data_target_iter = iter(train_target)
        model_path = ""

        for i in range(len_dataloader):

            step = epoch * len_dataloader + i

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = next(data_source_iter)
            s_img, s_label = data_source

            # s_grid = make_grid(s_img, nrow=6, normalize=True)
            # writer.add_image("soruce batch images", s_grid, global_step=step)

            my_net.zero_grad()

            batch_size = s_label.size(0)

            domain_label = torch.zeros(batch_size).long() # set to 0 as it belongs to source domain 

            s_img = s_img.to(device)
            s_label = s_label.to(device)
            domain_label = domain_label.to(device)

            class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label) # classification loss
            err_s_domain = loss_domain(domain_output, domain_label) # source domain loss

            # training model using target data
            data_target = next(data_target_iter)
            t_img, _ = data_target

            # t_grid = make_grid(t_img, nrow=6, normalize=True)
            # writer.add_image("Target batch images", t_grid, global_step=step)

            batch_size = t_img.size(0)

            domain_label = torch.ones(batch_size).long() # set to 1 as it belongs to target domain

            t_img = t_img.to(device)
            domain_label = domain_label.to(device)

            _, domain_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label) # target domain loss
            err = err_t_domain + err_s_domain + err_s_label # the total loss for the current batch is computed as the sum of the source domain loss, target domain loss, and classification loss.

            err.backward()
            optimizer.step()

            pred_class = class_output.argmax(dim=1)
            correct_class_batch += pred_class.eq(s_label).sum().item()
            total_samples_batch += s_label.size(0)
            
            if i % 10 == 0:
                batch_accuracy =  correct_class_batch / total_samples_batch
                writer.add_scalar("Batch accuracy", batch_accuracy, global_step=step)
                logging.info("Batch: {0}\t source domain loss: {1:.4f}\t target domain loss: {2:.4f}\t Combined Loss: {3:.4f}\t Batch Accuracy: {4:.4f}\r".format(i, err_s_domain.item(), err_t_domain.item(), err.item(), batch_accuracy))

            model_path = '{0}/nwpu_aid_epoch_current.pth'.format(model_root)
            torch.save(my_net, model_path)
        
        epoch_accuracy = correct_class_batch / total_samples_batch
        writer.add_scalar("Epoch accuracy ", epoch_accuracy, global_step=epoch)
        logging.info("Epoch: {0}\t Epoch Accuracy: {1:.4f}\r".format(epoch, epoch_accuracy))
                
        correct_class_epoch += correct_class_batch
        total_samples_epoch += total_samples_batch

        # Reset batch accuracy variables for the next epoch
        correct_class_batch = 0
        total_samples_batch = 0

        if epoch % 5 == 0 or epoch != 0:
            accu_s = test(val_source, model_path)
            writer.add_scalar("Test accouracy source", accu_s, global_step=epoch)
            logging.info("Accuracy of the {0} dataset: {1:.4f}".format(source_dataset_name, accu_s))
            accu_t = test(val_target, model_path)
            writer.add_scalar("Epoch accuracy", accu_t, global_step=epoch)
            logging.info("Accuracy of the {0} dataset: {1:.4f}".format('AID', accu_t))

        if accu_t > best_accu_t:
            best_accu_t = accu_t
            best_epoch = epoch
            torch.save(my_net, '{0}/{1}_{2}_{3}.pth'.format(model_root, experiment_name,  best_accu_t, best_epoch))
    writer.close()