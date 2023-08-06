import glob
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def generate_paths(path_expression, path_filename):

    all_path = glob.glob(path_expression)
    with  open(path_filename, "a") as file:
        for path in all_path:
            file.write(path + "\n")
        file.close

def show_images_from_dataloader(dataloader, num_images=25, nrow=5, title=None, figsize=(10, 10)):
    """
    Function to show images from a PyTorch DataLoader in a grid format.
    show_images_from_dataloader(train_target, num_images=25, nrow=5, title="Sample Images from DataLoader")
    """
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images[:num_images]

    plt.figure(figsize=figsize)
    plt.axis("off")
    if title is not None:
        plt.title(title)

    grid = vutils.make_grid(images, nrow=nrow, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
