from typing import Callable, Any
from pathlib import Path

import shutil
import tarfile 
import requests
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio

from torchdata.datapipes.iter import IterDataPipe, Zipper, IterableWrapper
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder 
import torchvision.transforms.v2 as t

from lightning import LightningDataModule

from hyperparameters import Hyperparameters

class ImageDataLoader(IterDataPipe):
    def __init__(self, 
                 src_dp: IterDataPipe, 
                 label_encoder: LabelEncoder,
                 transform: Callable | None = None):
        self.src_dp  = src_dp 
        self.le = label_encoder
        self.transform = transform if transform else self._default_transform
    
    def __iter__(self): 
        for path, label in self.src_dp:
            image = self._load_image(path)
            #image = self._minmax_image(image)
            image = self.transform(image) #type: ignore
            label = self._encode_label(label)
            yield (image, label)
     
    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = (iio.imread(uri = image_path,
                           plugin = "pillow",
                           extension = ".jpg")
                    .squeeze())

        #Duplicate Grayscale Image
        if image.ndim == 2:
            image = np.stack((image,)*3, axis = -1)
        assert image.shape[-1] == 3, "Not A 3 Channel Image"
        return image

    def _encode_label(self, label) -> torch.Tensor:
        return torch.tensor(
            self.le.transform([label])[0], #type: ignore
        dtype = torch.long)
    
    def _minmax_image(self, image: torch.Tensor) -> torch.Tensor:
        return (image - image.min()) / (image.max() - image.min())
    
    def _default_transform(self, image: torch.Tensor | np.ndarray) -> torch.Tensor:
        return t.Compose([
            t.ToImage(),
            t.ToDtype(torch.float32, scale=True),
            t.Resize((256, 256), antialias=True),
        ])(image)

class ImagenetteDataModule(LightningDataModule):
    def __init__(self, root: Path, params: Hyperparameters, transform: Callable | None = None) -> None:
        super().__init__()
        self.root = root
        if not self.root.is_dir():
            self.root.mkdir(parents = True)
        self.transform = transform
        self.batch_size = (params.batch_size // params.grad_accum)

        #TODO: Figure out how to automate getting num_workers
        #os.cpu_count or something like that
        self.num_workers = params.num_workers

        self.save_hyperparameters(params.get_datamodule_dict(),
            ignore = ["transform", "params"])

    def prepare_data(self) -> None:
        if self._is_empty_dir(self.root):
            url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"  
            print("Root is Empty, Downloading Dataset")
            archive: Path = self.root / "archive.tgz"
            self._download_from_url(url, archive)
            print("Extracting Dataset")
            self._extract_tgz(archive, self.root)
            print("Deleting Archive")
            archive.unlink(missing_ok=True)
            print("Moving Items to Root")
            self._move_dir_up(self.root / "imagenette2")
            print("Done!")
    
    def setup(self, stage: str) -> None:
        self._setup_local()
        if stage == "fit":
            self.train_dataset = self._prepare_local_train()
            self.val_dataset = self._prepare_local_val() 
        
        elif stage == "validate":
            self.val_dataset = self._prepare_local_val()

        elif stage == "test":
            self.val_dataset = self._prepare_local_val()

        elif stage == "predict":
            self.val_dataset = self._prepare_local_val()
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.train_dataset, 
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            #persistent_workers = True,
            pin_memory = True,
            shuffle = True
            )

    def val_dataloader(self) -> DataLoader:
        #NOTE: if setting num workers, make sure
        #      the entire dataset is returned only once 
        return DataLoader(
            dataset = self.val_dataset, 
            batch_size = self.batch_size,
            )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_dataset, 
            batch_size = self.batch_size,
            )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset = self.val_dataset, 
            batch_size = self.batch_size,
            )

    def _setup_local(self) -> None:
        df = pd.read_csv(self.root/"noisy_imagenette.csv")
        df = df[["path", "is_valid"]]
        df["path"] = df["path"].apply(lambda x: self.root / x)
        df["label"] = df["path"].apply(lambda x: x.parent.stem)

        self.val_df = (df[df["is_valid"] == True]
                .drop(columns=["is_valid"])
                .sort_values("label")
                .reset_index(drop=True))

        self.train_df = (df[df["is_valid"] == False]
                .drop(columns=["is_valid"])
                .sort_values("label")
                .reset_index(drop=True))

        self._prepare_label_encoder(df["label"].unique())

    def _prepare_local_train(self) -> Any:
        pipe = self._datapipe_from_dataframe(self.train_df)
        pipe = (pipe 
                    .shuffle(buffer_size=len(self.train_df))
                    #.sharding_filter()
                    #.prefetch()
                    #.pinned_memory()
                    #.load_image_data()
                    #.set_length()
                )
        pipe = ImageDataLoader(pipe, self.label_encoder, self.transform) #type:ignore 
        pipe = pipe.prefetch(self.batch_size)
        pipe = pipe.set_length(len(self.train_df))
        return pipe

    def _prepare_local_val(self) -> Any:
        pipe = self._datapipe_from_dataframe(self.val_df)
        pipe = ImageDataLoader(pipe, self.label_encoder, self.transform) #type: ignore
        pipe = pipe.set_length(len(self.val_df))
        return pipe

    def _datapipe_from_dataframe(self, dataframe: pd.DataFrame) -> Any:
        return Zipper(
            IterableWrapper(dataframe.path),
            IterableWrapper(dataframe.label)
            )
    
    def _prepare_label_encoder(self, class_names: list) -> None:
        self.label_encoder = LabelEncoder().fit(sorted(class_names))

    def _download_from_url(self, url: str, local_filename: Path) -> None:
        response = requests.head(url)
        file_size = int(response.headers.get("Content-Length", 0))

        with requests.get(url, stream=True) as response:
            with open(local_filename, "wb") as output_file:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, unit_divisor=1024
                ) as progress_bar:
                    for data in response.iter_content(chunk_size=1024*1024):
                        output_file.write(data)
                        progress_bar.update(len(data))
    
    def _extract_tgz(self, tgz_file, out_dir) -> None: 
        with tarfile.open(tgz_file, "r:gz") as tar:
            tar.extractall(out_dir)
        
    def _is_empty_dir(self, path: Path) -> bool:
        return not list(path.iterdir())
        
    def _move_dir_up(self, source_dir: Path) -> None:
        for path in source_dir.iterdir():
            dest_path = source_dir.parent / path.name
            if path.is_dir():
                path.rename(dest_path)
            else:
                shutil.move(path, dest_path)
        source_dir.rmdir()

def viz_batch(batch: tuple[torch.Tensor, torch.Tensor], le: LabelEncoder, df: pd.DataFrame) -> None:
    images, targets = batch
    labels = le.inverse_transform(targets.ravel())
    labels = [df.loc[x].label for x in labels]
    assert images.shape[0] == targets.shape[0], "#images != #targets"

    subplot_dims:tuple[int, int]
    if images.shape[0] <= 8:
        subplot_dims = (1, images.shape[0])
    else:
        subplot_dims = (int(np.ceil(images.shape[0]/8)), 8)

    figsize = 20
    figsize_factor = subplot_dims[0] / subplot_dims[1]
    _, axes = plt.subplots(nrows = subplot_dims[0], 
                           ncols = subplot_dims[1], 
                           figsize = (figsize, figsize * figsize_factor))
    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(images[idx].permute(1, 2, 0))
        ax.tick_params(axis = "both", which = "both", 
                       bottom = False, top = False, 
                       left = False, right = False,
                       labeltop = False, labelbottom = False, 
                       labelleft = False, labelright = False)
        ax.set_xlabel(f"{labels[idx]}({targets[idx].item()})")