import os
import torch.backends.cudnn as cudnn
import torch.utils.data


def test(test_dataloader, model_path):

    root =  os.path.dirname(__file__)
    cudnn.benchmark = True
    batch_size = 32
    alpha = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """ test """

    my_net = torch.load(os.path.join(
        root, model_path
    ))
    my_net = my_net.eval()

    my_net = my_net.to(device)

    len_dataloader = len(test_dataloader)
    data_target_iter = iter(test_dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = t_label.size(0)

        t_img = t_img.to(device)
        t_label = t_label.to(device)

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.argmax(dim=1)
        n_correct += pred.eq(t_label).sum().item()
        n_total += batch_size

        i += 1

    accu = n_correct / n_total

    return accu 