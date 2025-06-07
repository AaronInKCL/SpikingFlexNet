#!/Users/donyin/miniconda3/envs/imperial/bin/python
"""
This class:
1. takes the folder and reads the configuration.json file
2. construct the model and load the checkpoint
- this serves as a base class for all the analysis scripts
- in other words, it loads the model and the checkpoint

e.g., loader = RunLoader(some_path)

you get:
        loader.model
        loader.optimizer
        loader.logger
        loader.current_epoch
        loader.current_loss
        loader.criterion

"""

import json, torch, os
from torch import nn
from rich import print
from torch import optim
from pathlib import Path
from natsort import natsorted
from src.utils.general import banner
from torch.utils.data import DataLoader
from src.utils.device import select_device
from src.utils.hyperparams.settings import AttrDict
from src.utils.general import apply_kaiming_initialization
from src.training.dataset_select import get_dataset_obj
from src.training.dataset_subset import create_random_subset
from MetricDB import MetricDB

from src.modules import models  # e.g., getattr(models, "SimpleFlexNet") / VGG16


class RunLoader:
    def __init__(
        self,
        run_folder: Path,
        whether_load_checkpoint: bool = True,
        whether_instantiate_model=True,
    ):
        # regular stuff
        self.device, self.run_folder = select_device(), Path(run_folder)

        # for copying
        self.whether_load_checkpoint = whether_load_checkpoint

        self._load_config()

        if whether_instantiate_model:
            self._init_plain_model()
            self._init_checkpoint()

        if whether_instantiate_model and whether_load_checkpoint:
            banner("Loading Checkpoint")
            self._load_model_and_optimizer()

        self.logger = MetricDB(datafile_dir=self.run_folder / "logs", verbose=True)

    # ---- loading model ----
    def _load_config(self):
        configurations_dir = self.run_folder / "configurations.json"
        self.config = json.load(configurations_dir.open("r"))
        self.config = AttrDict(self.config)
        return self

    def _init_plain_model(self):
        self.model = getattr(models, self.config.get("network"))(config=self.config)
        apply_kaiming_initialization(self.model)
        self.model, learning_rate = self.model.to(self.device), self.config.get("learning_rate")
        match self.config.get("optimizer", "SGD"):
            case "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
            case "ADAM":
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
            case _:
                raise NotImplementedError
        self.current_epoch, self.current_loss = 0, 0
        self.criterion = nn.CrossEntropyLoss()  # assume that the dataset is balanced
        return self

    # ---- init ----
    def _do_a_dummy_backward_pass(self):  # [IMPORTANT]: make dataset input
        """this is useful when plotting the gradients as well as debuggin model architecture"""
        torch.manual_seed(42)
        dataset_name = self.config.get("dataset")
        dataset = get_dataset_obj(dataset_name, "TRAIN")
        dataset = create_random_subset(dataset, self.config.get("batch_size"))
        train_loader = DataLoader(dataset, batch_size=self.config.get("batch_size"), shuffle=True)
        images, labels = next(iter(train_loader))
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self.model(images)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        loss.backward()
        return self

    def _do_a_dummy_forward_pass(self):
        """[IMPORTANT]: run a dummy forward pass to initialise the model first"""
        torch.manual_seed(42)
        self.model.train()
        self.model(torch.rand(1, *self.config.in_dimensions).to(self.device))
        self.model.eval()
        return self

    # ---- save and load ----
    def _init_checkpoint(self):
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        if not list(ckpt_dir.glob("*.pth")):
            self.save_checkpoint()

    def _load_model_and_optimizer(self):
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_files = list(ckpt_dir.glob("*.pth"))
        ckpt_files = natsorted(ckpt_files)

        if not len(ckpt_files) == 1:
            banner("[WARNING]")
            print(f"Expecting 1 .pth file, found {len(ckpt_files)}.")
            print(f"run_folder: {self.run_folder}")
            assert False, "Check the run folder"

        save_content = torch.load(ckpt_files[0], map_location=self.device)
        self.model.load_state_dict(save_content["model_state_dict"])
        self.optimizer.load_state_dict(save_content["optimizer_state_dict"])
        self.current_epoch, self.current_loss = save_content["epoch"], save_content["loss"]
        return self

    def save_checkpoint(self):
        """
        save: model (state_dict); optimizer (state_dict); epoch number; loss
        """
        ckpt_dir = self.run_folder / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        existing = ckpt_dir.glob("*.pth")
        existing = [f for f in existing if f.name.startswith("checkpoint")]
        [os.remove(f) for f in existing]

        save_name = self.current_epoch
        save_content = {"model_state_dict": self.model.state_dict()}
        save_content.update({"optimizer_state_dict": self.optimizer.state_dict()})
        save_content.update({"epoch": self.current_epoch, "loss": self.current_loss})
        torch.save(save_content, ckpt_dir / f"checkpoint_{save_name}.pth")


if __name__ == "__main__":
    pass
