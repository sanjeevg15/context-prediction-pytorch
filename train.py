from torchvision.models import AlexNet
import torchvision
from torch.optim import Adam
import torch.nn as nn
from context_predicton_trainer import ContextPredictionTrainer

caltech101_dataset = torchvision.datasets.Caltech101(root=".", download=True)

feature_extractor = AlexNet().features
classifier = nn.Sequential(nn.Flatten(), nn.Linear(2048, 8))

training_args={"optimizer": Adam, "num_epochs": 1000, "batch_size": 256, "patch_size": 40, "patches_gap": 7}
trainer = ContextPredictionTrainer(training_args)
trainer.train(feature_extractor=feature_extractor, classifier=classifier, dataset=caltech101_dataset, training_args=training_args, learning_rate=3e-4)

trainer = ContextPredictionTrainer(training_args=training_args)

trainer.train(feature_extractor=feature_extractor, classifier=classifier, dataset=caltech101_dataset, training_args=training_args, learning_rate=3e-4)