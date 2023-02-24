import time
import torch.utils.data
#import nvidia.dali.ops as ops
#import nvidia.dali.types as types
import torchvision.datasets as datasets
#from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
#from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

import torchvision.transforms as transforms



class rgb2hsvTRANSFORM(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        return self.rgb2hsv(img)

    def rgb2hsv(self, input, epsilon=1e-10):
        assert (input.shape[0] == 3)

        r, g, b = input[0, :, :], input[1, :, :], input[2, :, :]
        max_rgb, argmax_rgb = input.max(0)
        min_rgb, argmin_rgb = input.min(0)

        max_min = max_rgb - min_rgb + epsilon

        #print(max_min.shape)
        #print(g.shape)
        #print(r.shape)

        h1 = 60.0 * (g - r) / max_min + 60.0
        h2 = 60.0 * (b - g) / max_min + 180.0
        h3 = 60.0 * (r - b) / max_min + 300.0

        h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
        s = max_min / (max_rgb + epsilon)
        v = max_rgb
        out = torch.stack((h, s, v), dim=0)
        #print(out.shape)
        return out


def cifar_transform(is_training=True):
    if is_training:
        transform_list = [
                          # DATA AUGMENT
                          transforms.RandomHorizontalFlip(),
                          transforms.Pad(padding=4, padding_mode='reflect'),
                          transforms.RandomCrop(32, padding=0),

                          #transforms.RandomRotation(10),  # Rotates the image to a specified angel
                          #transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                          # Performs actions like zooms, change shear angles.
                          #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                          # transforms.Resize(42),
                          #transforms.CenterCrop(42),
                          transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          #rgb2hsvTRANSFORM()
                          ]
    else:
        transform_list = [#transforms.Resize(50),
                           # transforms.CenterCrop(42),
                            transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          #rgb2hsvTRANSFORM()
                          ]

    transform_list = transforms.Compose(transform_list)
    return transform_list


def imgnet_transform(is_training=True):
    if is_training:
        transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ColorJitter(brightness=0.5,
                                                                    contrast=0.5,
                                                                    saturation=0.3),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    else:
        transform_list = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    return transform_list


from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



"""
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                           world_size=1,
                           local_rank=0):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                    data_dir=image_dir + '/train',
                                    crop=crop, world_size=world_size, local_rank=local_rank)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size, auto_reset=True)
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                data_dir=image_dir + '/val',
                                crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size, auto_reset=True)
        return dali_iter_val


def get_imagenet_iter_torch(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                            world_size=1, local_rank=0):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                 pin_memory=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
                                                 pin_memory=True)
    return dataloader"""
