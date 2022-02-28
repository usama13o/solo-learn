from bisect import bisect_right
import h5py
import torch.utils.data as data
import PIL.Image as Image
import numpy as np
import datetime

from os import listdir
from os.path import join
from os.path import basename
import random

class wss_dataset(data.Dataset):
    def __init__(self, root_dir, split, transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(wss_dataset, self).__init__()
        #train dir 
        img_dir = root_dir

        self.image_filenames  = sorted([join(img_dir, x) for x in listdir(img_dir) if is_image_file(x)])
        self.target_filenames = [list(map(int,[x.split('-')[-1][:-4][1],x.split('-')[-1][:-4][4],x.split('-')[-1][:-4][7]])) for x in self.image_filenames]
        sp= self.target_filenames.__len__()
        sp= int(train_pct *sp)
        random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        elif split =='all':
            self.image_filenames = self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
        assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} patches'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        target = self.target_filenames[index]
        if sum(target) == 2:
            target = 3
        else:
            target = np.array(target).argmax()
        # load the nifti images
        if not self.preload_data:
            input  = Image.open(self.image_filenames[index])
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        if self.transform:
            input = self.transform(input)

        return input, target

    def __len__(self):
        return len(self.image_filenames)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz",'png','tiff','jpg',"bmp"])

def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
    filename: Name of the image file.
    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image
def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    return array



class Whole_Slide_Bag(data.Dataset):
    def __init__(self,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,
        ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained=pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        # self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
        
        return Image.fromarray(img)


class svs_h5_dataset(data.Dataset):
    def find_bin(self,y):
        l = [0]
        for ll,x in zip(l,self.tot):
            l.append(x+ll)
            where = list(map((lambda x: x-y ),l))
            which = bisect_right(where,0)
        return which
    def __init__(self, root_dir, split="all", transform=None, preload_data=False,train_pct=0.8,balance=True):
        super(svs_h5_dataset,self).__init__()
        #train dir 
        img_dir = root_dir

        self.image_filenames  = sorted([join(img_dir, x) for x in listdir(img_dir) if ".h5" in x ])

        # get total patches in each WSI
        tot=[]
        for can in range(len(self.image_filenames)):
            fn = self.image_filenames[can]
            tot.append(len(Whole_Slide_Bag(fn)))
        self.tot = tot
        
        self.target_filenames = []
        sp= self.target_filenames.__len__()
        sp= int(train_pct *sp)
        random.shuffle(self.image_filenames)
        if split == 'train':
            self.image_filenames = self.image_filenames[:sp]
        elif split =='all':
            self.image_filenames = self.image_filenames
        else:
            self.image_filenames = self.image_filenames[sp:]
            # find the mask for the image
        #assert len(self.image_filenames) == len(self.target_filenames)

        # report the number of images in the dataset
        print('Number of {0} images: {1} svs'.format(split, self.__len__()))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [open_image_np(ii)[0] for ii in self.image_filenames]
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        target = 0 #self.target_filenames[index]
        # get which Image 
        
        where = self.find_bin(index) - 1
        try:
            input = Whole_Slide_Bag(self.image_filenames[where])
        except:
            print(f"Couldnt find Image with index {where} with input index of {index}")

        # Which index in that image         
        which = index & len(input) - 1


        # load the nifti images
        if not self.preload_data:
            try:
                input = input[which]
            except:
                print(f"Couldn't find patch with index {which} in SVS with total of {len(input)} patches")
        else:
            input = np.copy(self.raw_images[index])

        # handle exceptions
        if self.transform:
            input = self.transform(input)

        return input, target

    def __len__(self):
        return sum(self.tot)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz",'png','tiff','jpg',"bmp"])

def open_image(filename):
    """
    Open an image (*.jpg, *.png, etc).
    Args:
    filename: Name of the image file.
    returns:
    A PIL.Image.Image object representing an image.
    """
    image = Image.open(filename)
    return image
def open_image_np(path):
    im = open_image(path)
    array = np.array(im)
    return array