import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import BertTokenizer
from datasets import load_from_disk, load_dataset


class NLPDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = torch.tensor(self.encodings['input_ids'][idx])
        label = torch.tensor(self.labels[idx])
        return item, label

    def __len__(self):
        return len(self.labels)


DATASETS = {
    'CIFAR10': {
        'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2471, 0.2435, 0.2616), 'num_classes': 10, 
        'image_channels': 3, 'size': 32, "download": True, "train_path": None, "test_path": None
    },
    'CIFAR100': {
        'mean': (0.5071, 0.4865, 0.4409), 'std': (0.2673, 0.2564, 0.2762), 'num_classes': 100, 
        'image_channels': 3, 'size': 32, "download": True, "train_path": None, "test_path": None
    },
    'SVHN': {
        'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970), 'num_classes': 10, 
        'image_channels': 3, 'size': 32, "download": True, "train_path": None, "test_path": None
    },
    'MNIST': {
        'mean': (0.5,), 'std': (0.5,), 'num_classes': 10, 'image_channels': 1, 
        'size': 28, "download": True, "train_path": None, "test_path": None
    },
    'IMAGENET': {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 
        'num_classes': 1000, 'image_channels': 3, 'size': 224, "download": False, 
        "train_path": "/data/liguanchen/Datasets/Imagenet1K/train/", 
        "test_path": "/data/liguanchen/Datasets/Imagenet1K/val/"
    },
    'IMAGENETTE': {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 
        'num_classes': 10, 'image_channels': 3, 'size': 224, "download": False, 
        "train_path": "/data/liguanchen/Datasets/imagenette2-320/train/", 
        "test_path": "/data/liguanchen/Datasets/imagenette2-320/val/"
    },
    'IMAGEWOOF': {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 
        'num_classes': 10, 'image_channels': 3, 'size': 224, "download": False, 
        "train_path": "/data/liguanchen/Datasets/imagewoof2-320/train/", 
        "test_path": "/data/liguanchen/Datasets/imagewoof2-320/val/"
    },
    'TINY-IMAGENET': {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 
        'num_classes': 200, 'image_channels': 3, 'size': 64, "download": False, 
        "train_path": "/data/liguanchen/Datasets/tiny-imagenet-200/train/", 
        "test_path": "/data/liguanchen/Datasets/tiny-imagenet-200/val/"
    },
    'MINI-IMAGENET': {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 
        'num_classes': 100, 'image_channels': 3, 'size': 224, "download": False, 
        "train_path": "/data/liguanchen/Datasets/mini-imagenet/train/", 
        "test_path": "/data/liguanchen/Datasets/mini-imagenet/val/"
    },
    'IMAGENET-LT': {
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 
        'num_classes': 1000, 'image_channels': 3, 'size': 224, "download": False, 
        "train_path": "/data/liguanchen/Datasets/ImageNet_LT/train/", 
        "test_path": "/data/liguanchen/Datasets/ImageNet_LT/val/"
    },
    'SST2': {'num_classes': 2},
    'IMDB': {'num_classes': 2},
}


def build_transform(is_train, args):   # for 224x224 size images only
    do_resize = DATASETS[args.dataset_name]['size'] > 64
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    input_size = (DATASETS[args.dataset_name]['image_channels'], DATASETS[args.dataset_name]['size'], DATASETS[args.dataset_name]['size'])

    if is_train:
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            mean=mean,
            std=std,
        )
        if not do_resize:
            transform.transforms[0] = transforms.RandomCrop(input_size[-1], padding=4)
        return transform

    t = []
    if do_resize:
        if DATASETS[args.dataset_name]['size'] >= 384:  
            t.append(transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC))
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(DATASETS[args.dataset_name]['size'] / args.crop_pct)
            t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
            t.append(transforms.CenterCrop(DATASETS[args.dataset_name]['size']))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_dataset(args):
    if args.dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', "SVHN"]:
        train_padding = 4 if DATASETS[args.dataset_name]['size'] == 32 else (32 - DATASETS[args.dataset_name]['size'] ) // 2
        test_padding = 0 if DATASETS[args.dataset_name]['size'] == 32 else (32 - DATASETS[args.dataset_name]['size'] ) // 2
        assert (test_padding * 2 + DATASETS[args.dataset_name]['size']) == 32

        min_size = 28 if args.dataset_name == 'MNIST' else 32

        if 'CIFAR10' == args.dataset_name or 'CIFAR100' == args.dataset_name:
            transform_train = transforms.Compose([
                transforms.RandomCrop(min_size, padding=train_padding),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(DATASETS[args.dataset_name]['mean'], DATASETS[args.dataset_name]['std']),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(min_size, padding=train_padding),
                transforms.ToTensor(),
                transforms.Normalize(DATASETS[args.dataset_name]['mean'], DATASETS[args.dataset_name]['std']),
            ])

        transform_test = transforms.Compose([
            transforms.RandomCrop(min_size, padding=test_padding),
            transforms.ToTensor(),
            transforms.Normalize(DATASETS[args.dataset_name]['mean'], DATASETS[args.dataset_name]['std']),
        ])

        if 'SVHN' == args.dataset_name:
            trainset = getattr(torchvision.datasets, args.dataset_name)(
                root='./data-' + args.dataset_name, split='train', 
                download=DATASETS[args.dataset_name]['download'], transform=transform_train
            )
        else:
            trainset = getattr(torchvision.datasets, args.dataset_name)(
                root='./data-' + args.dataset_name, train=True, 
                download=DATASETS[args.dataset_name]['download'], transform=transform_train
            )

        train_sample_num = int(len(trainset))
        trainset, _ = torch.utils.data.random_split(trainset, [train_sample_num, len(trainset) - train_sample_num])
        
        if 'SVHN' == args.dataset_name:
            testset = getattr(torchvision.datasets, args.dataset_name)(
                root='./data-' + args.dataset_name, split='test', 
                download=DATASETS[args.dataset_name]['download'], transform=transform_test
            )
        else:
            testset = getattr(torchvision.datasets, args.dataset_name)(
                root='./data-' + args.dataset_name, train=False, 
                download=DATASETS[args.dataset_name]['download'], transform=transform_test
            )

        return trainset, testset
    
    elif args.dataset_name in ['IMAGENETTE', 'IMAGENET', 'IMAGEWOOF', 'MINI-IMAGENET', 'TINY-IMAGENET', 'IMAGENET-LT']:
        train_transform = build_transform(True, args)
        val_transform = build_transform(False, args)
        train_data_root = DATASETS[args.dataset_name]['train_path']
        val_data_root = DATASETS[args.dataset_name]['test_path']
        train_dataset = ImageFolder(train_data_root, transform=train_transform)
        val_dataset = ImageFolder(val_data_root, transform=val_transform)
        return train_dataset, val_dataset

    elif args.dataset_name in ['SST2', 'IMDB']:
        try:
            tokenizer = BertTokenizer.from_pretrained('/data/liguanchen/modelzoo/bert-base-uncased')
            dataset = load_from_disk("/data/liguanchen/Datasets/sst2") if args.dataset_name == "SST2" else load_from_disk("/data/liguanchen/Datasets/imdb")
        except:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            dataset = load_dataset("glue", "sst2") if args.dataset_name == "SST2" else load_dataset("imdb")
        if args.dataset_name == "SST2":
            eval_name = "validation"
            def encode(examples):
                return tokenizer(examples['sentence'], 
                                truncation=True, 
                                padding='max_length', 
                                max_length=256)
        else:
            eval_name = "test"
            def encode(examples):
                return tokenizer(examples['text'], 
                                truncation=True, 
                                padding='max_length', 
                                max_length=256)
        encoded_train = encode(dataset['train'])
        encoded_val = encode(dataset[eval_name])
        train_dataset = NLPDataset(encoded_train, dataset['train']['label'])
        val_dataset = NLPDataset(encoded_val, dataset[eval_name]['label'])
        return train_dataset, val_dataset
    else:
        raise ValueError("Unknown Dataset!")
