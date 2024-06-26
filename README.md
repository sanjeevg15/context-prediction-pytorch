# context-prediction-pytorch
PyTorch Implementation of [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192)

Usage Example: 

```
training_args={"optimizer": Adam, "num_epochs": 1000, "batch_size": 256, "patch_size": 40, "patches_gap": 7}
trainer = ContextPredictionTrainer(training_args)
trainer.train(feature_extractor=feature_extractor, classifier=classifier, dataset=caltech101_dataset, training_args=training_args, learning_rate=3e-4)
