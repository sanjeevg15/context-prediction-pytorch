import torch
from torch.nn import Module
from typing import Tuple, Union
from torchvision.models import AlexNet
import torchvision
import random
import cv2
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from tqdm.notebook import tqdm
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler



class ContextPredictionTrainer:
    def __init__(self, training_args=None):
        super().__init__()
        self.jitter_range = [3, 7]
        self.patches_gap = 48
        self.loss_fn = CrossEntropyLoss()
        self.patch_size = 96
        self.image_size = 400
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.labels_to_offset = {
            0: (-1, -1),
            1: (0,  -1),
            2: (1, -1),
            3: (1, 0),
            4: (1, 1),
            5: (0,  1),
            6: (-1,1),
            7: (-1,  0),
        }

        self.training_args = training_args or {}

        self._default_training_args = self.get_default_training_args()

        for k, v in self._default_training_args.items():
            if k not in self.training_args:
                self.training_args[k] = v

    def get_default_training_args(self) -> dict:
        default_training_args = {
            "batch_size": 32,
            "num_epochs": 2
        }
        return default_training_args


    def train(self, feature_extractor, classifier, dataset: Dataset, training_args: dict, learning_rate=0.001, resume_run_id: str=None):

        num_epochs = training_args["num_epochs"]
        optimizer = Adam(params={*classifier.parameters(), *feature_extractor.parameters()}, lr=learning_rate)
        num_iters_per_epoch = len(dataset)//training_args["batch_size"]

        train_dataset = self._get_context_prediction_dataset(dataset)
        train_loader = DataLoader(train_dataset, batch_size=training_args["batch_size"], shuffle=True)

        feature_extractor = feature_extractor.to(self.device)
        classifier = classifier.to(self.device)
        # Start the training loop

        if resume_run_id is not None:
            run = wandb.init(project="context-prediction", id=resume_run_id, resume="must")
            num_epochs_completed = run._step//num_iters_per_epoch
            epochs_range = range(num_epochs_completed+1, num_epochs)
        else:
            run = wandb.init(project="context-prediction", config=training_args)
            epochs_range = range(num_epochs)


        feauture_extractor_graph = wandb.watch(feature_extractor, log="all", log_freq=5, idx=0, log_graph=True)
        classifier_graph = wandb.watch(classifier, log="all", log_freq=5, idx=1, log_graph=True)


        for epoch in tqdm(epochs_range):

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for _, (center_patch, offset_patch, label, _, _, _) in enumerate(pbar):
                fp = list(feature_extractor.parameters())
                cp = list(classifier.parameters())


                center_patch = center_patch.to(self.device)
                offset_patch = offset_patch.to(self.device)
                label = label.to(self.device)

                center_patch_features = feature_extractor(center_patch)
                offset_patch_features = feature_extractor(offset_patch)

                concat_features = torch.cat([center_patch_features, offset_patch_features], dim=1)

                pred = classifier(concat_features)
                loss = self.loss_fn(pred, label)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                pbar.set_postfix_str(f" loss: {loss.item()}")
                # Log the loss to wandb
                data = {
                    "loss": loss.item(),
                    "feature_extractor": feauture_extractor_graph,
                    "classifier": classifier_graph,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                }
                run.log(data, commit=True)


            run.log({"epoch": epoch}, commit=True)

            ckpt_path_classifier = f"context_prediction_trainer_epoch_{epoch}_classifier.pt"
            torch.save(classifier.state_dict(), ckpt_path_classifier)

            ckpt_path_feature_extractor = f"context_prediction_trainer_epoch_{epoch}_feature_extractor.pt"
            torch.save(feature_extractor.state_dict(), ckpt_path_feature_extractor)

            if epoch % 10 == 0:
                run.save(ckpt_path_classifier, policy="live")
                run.save(ckpt_path_feature_extractor, policy="live")

        run.finish()


    def _get_context_prediction_dataset(self, dataset: Dataset):

        trainer = self
        # Ensure image size is not too small

        # image, _ = dataset[0]
        # h, w = image.size

        # if h < 400 or w < 400:
            # raise ValueError("Image should be at least 400 pixels in both dimensions")

        class ContextPredictionDataset(Dataset):
            def __init__(self, labeled_dataset):
                self.labeled_dataset = labeled_dataset

            def __len__(self) -> int:
                return len(self.labeled_dataset)

            def __getitem__(self, index):
                # index = index%3
                image, _ = self.labeled_dataset[index]
                image =image.resize((trainer.image_size, trainer.image_size))
                h, w = image.size
                image_np = np.array(image)/255.0
                if len(image_np.shape) == 2:
                    image_np = np.stack([image_np]*3, axis=2)

                image = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
                label = torch.randint(low=0, high=8, size=(1,)).item()
                offset = trainer.labels_to_offset[label]
                center = (h//2, w//2)
                center_patch, box_center = trainer.get_patch(image, center, patch_size=trainer.patch_size)
                offset_patch, offset_center, box_offset = trainer.get_offset_patch(image, label, patch_size=trainer.patch_size)


                return (center_patch, offset_patch, label, box_center, box_offset, offset_center)

        return ContextPredictionDataset(dataset)

    def get_patch(self, x: torch.Tensor, patch_center: Tuple[int, int], patch_size: int):
        _, h, w = x.shape

        half_patch_size = patch_size//2

        patch = x[:, patch_center[1] - half_patch_size: patch_center[1] + half_patch_size, patch_center[0] - half_patch_size: patch_center[0] + half_patch_size]

        box = ((patch_center[0] - half_patch_size, patch_center[1] - half_patch_size), (patch_center[0] + half_patch_size, patch_center[1] + half_patch_size))
        return patch, box


    def get_offset_patch(self, x: torch.Tensor, label, patch_size: int) -> torch.Tensor:
        """
            Returns a patch that is offset in the direction defined by the label

            Args:
                x (torch.Tensor): Input image of shape (C, H, W)
                label (int): The label of the patch
                patch_size (int): The size of the patch
            Returns:
                torch.Tensor: The offset patch
        """
        _, h, w = x.shape
        center_x = h//2
        center_y = w//2


        offset = self.labels_to_offset[label]

        offset_distance = self.patch_size + self.patches_gap
        offset_patch_center = (center_x + offset[0]*offset_distance, center_y + offset[1] * offset_distance)
        offset_patch_center = self.jitter_center(offset_patch_center)

        patch, box = self.get_patch(x, offset_patch_center, patch_size)
        return patch, offset_patch_center, box


    def jitter_center(self, center: Tuple[int, int]) -> Tuple[int, int]:
        x, y = center
        jitter_low, jitter_high = self.jitter_range
        jitter_x = random.randint(jitter_low, jitter_high + 1)
        jitter_y = random.randint(jitter_low, jitter_high + 1)

        if random.random() <= 0.5:
            jitter_x *= -1

        if random.random() <= 0.5:
            jitter_y *= -1

        return (x + jitter_x, y + jitter_y)

    def apply_color_projection(self, x):
        raise NotImplementedError

    def forward(self, x, mode="eval"):
        patch1, patch2 = None, None
        input = torch.cat([patch1, patch2])
        r1, r2 = self.backbone(input)
        pred = self.context_prediction_classifier(r1, r2)

        if mode=="train":
            loss = self.loss_fn(label, pred)
            return loss, label, pred, patch1, patch2
        else:
            return None, None, self.backbone(x), None, None

        if mode == "eval":
            pred = self.backbone(x)
            return pred

        elif mode == "train":
            center = [x.shape[0]//2, x.shape[1]//2]
            patch1 = self.get_patch(x, center, self.patch_size)
            patch2, offset_patch_center = self.get_offset_patch(x, label)
            r1 = self.backbone(patch1)
            r2 = self.backbone(patch2)
            pred = self.context_prediction_classifier(r1, r2)
            loss = self.loss_fn(label, pred)
            return loss, label, pred, patch1, patch2
        else:
            raise ValueError("Invalid mode, expected one of 'eval' or 'train', but got {}".format(mode))