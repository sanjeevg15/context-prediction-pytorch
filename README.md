# Context Prediction Pytorch
PyTorch Implementation of [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/pdf/1505.05192) by Carl Doersch, _et al_. 

This repository allows you to train _any_ "feature extractor" on _any_ image dataset in a self-supervised fashion using the Context Prediction methodology described in the above seminal research paper.

A feature extractor is used to convert images to meaningfully rich vector embeddings. A classifier is used to classify these embeddings into one of 8 categories. After the training, the classifier is discarded and the feature extractor can be used for _downstream_ tasks.

Usage Example: 

```
training_args={"optimizer": Adam, "num_epochs": 1000, "batch_size": 256, "patch_size": 40, "patches_gap": 7}
trainer = ContextPredictionTrainer(training_args)
trainer.train(feature_extractor=feature_extractor, classifier=classifier, dataset=caltech101_dataset, training_args=training_args, learning_rate=3e-4)
