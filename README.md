# AppliedAGI Assignment

---
## Environment

1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Set-up the following environment:
```bash
conda env create -f environment.yml -n my-env
conda activate my-env
```

---
## Dataset

I chose Cityscapes, https://www.cityscapes-dataset.com/, because it provides high-quality semantic labels of urban scenes.
I have never experimented with it before, but I thought it is related to AAGI's range of activities.

The dataset contains a total of `5000` images at a resolution of `2048x1024`.
Fine-grained semantic maps are available and are depicting `19` classes, e.g. car, bus, bike, wall, road, etc...

First, export the path where the dataset will be stored:
```bash
export CITYSCAPES_DATASET=/path/to/data/cityscapes
```
Simply create an account to proceed to the download, you will have to type your credentials:
```bash
cs Download -d $CITYSCAPES_DATASET gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
```

---
## Task 1
I was tasked to define an infinite dataloader from a finite source of data.

To do so, the problem was broken down into:
1. Defining a simple `FiniteDataset`, which fetches images from the disk.
2. Defining an `InfiniteSampler`, which defines an infinite generator, which will yield indexes from the dataset.
3. Passing those two objects to the existing `torch.utils.data.DataLoader`, in order to enable the multi-processing and pinning to memory, etc...

To test it:
```bash
python infinite_dataloader.py --data_root $CITYSCAPES_DATASET/leftImg8bit --batch_size 16 --num_workers 4
```

---
## Task 2
The task we want to solve is semantic segmentation on the Cityscapes dataset.

### Libraries
This codebase uses:
- `torch`, for the various computations and automatic differentiation.
- `lightning`, it is a high level framework built on top of `torch` to alleviate the need to write boilerplate code w.r.t logging, seamlessly enabling the use of multiple GPUs, etc.
- `torchmetrics`, it defines a `Metric` objects to validate the performance of the model, and scales to multiple GPUs.
- `wandb`, it is a logging library with a web interface, offering lots of insightful features for plotting the collected data.
- `timm`, which is a library centralising the major vision models and pre-trained checkpoints, ready to use for fine-tuning on downstream tasks.
- `albumentations`, which is a library offering many options for image data augmentations, such as colour jitter, random crops, etc...

### Choice of Model
I chose to use `SwinV2` (https://arxiv.org/abs/2111.09883) as the backbone, and `Feature Pyramid Network` (https://arxiv.org/abs/1612.03144) as the decoding head.

The reason why I chose a `SwinV2` over `ViT` is that it extracts feature maps at multiple scales, which is well suited for the semantic segmentation task.

For the decoding head, I went for a simple `Feature Pyramid Network` head, which just progressively upsamples and fuses the extracted maps, from the encoder, to produce high resolution semantically rich features.

I could have opted for more complex decoder heads, such as `Pyramid Scene Parsing` (https://arxiv.org/abs/1612.01105) or `Unified Perceptual Parsing` (https://arxiv.org/abs/1807.10221), but that might be much for an early prototyping task.

### Repository Overview:
Here is an overview of the codebase:
- `datamodule.py`: Contains the `CityscapesDataModule`, which is a convenience wrapper for the train and validation datasets, transforms and dataloaders.
- `transform.py`: Defines the transformations applied during training and evaluations, as well the data augmentations performed during training.
- `helpers.py`: Contains some helpful functions for converting the semantic maps to coloured maps, etc.
- `model.py`: Contains the `FPNSwinTransformerV2` semantic segmentation model.
- `lightning_model.py`: Contains the `LitSemanticSegmentationModel`, which defines the behaviour of the training and evaluation loops, the performance metrics and the logging of sample images at each epoch.
- `scheduler.py`: Defines the `CosineScheduler` which updates the learning rate as the training progress.
- `train.py`: Contains the main script to kickstart the training of the model, it behaves in a "smart" way in the sense that it automatically collects the amount of CPUs/GPUs available in the machine to distribute training automatically, optimise the number of data workers, as well as decide how to split the load between the GPUs and whether it is needed to accumulate gradients given the requested theoretical batch size and the machine capabilities, etc.

Each file has been commented, feel free to inspect the code and read the comments :)

### Training the Model
First, locate your experiment folder:
```bash
export AAGI_EXPERIMENT=/path/to/my/experiment
```
To train the model, and upload the results to `wandb`:
```bash
python train.py --experiment $AAGI_EXPERIMENT  --root $CITYSCAPES_DATASET --no-offline
```
This will read the `hparams.yaml` contained in `/path/to/my/experiment`, which looks like this:
```yaml
model:
  name: FPNSwinTransformerV2 # Name of the semantic segmentation model, in case we have several.
  backbone: swinv2_tiny_window8_256 # Name of the backbone for that model.
  pretrained: true # Whether to use a pretrained backbone. 
loss:
  label_smoothing: 0.0 # Label smoothing parameter for the cross entropy loss.
data:
  crop_size: [256, 256] # Size of the crop for the training data, in the case of transformer it is dependent on the chosen backbone.
optim:
  batch_size: 32 # Theoretical batch size.
  max_epochs: 15 # Number of training epochs.
  lr: 1e-4 # Base learning rate.
  wd: 1e-2 # L2 regularisation parameter.
scheduler:
  warm_up_epochs: 1 # Number of epochs of linear warm-up
  warm_up_factor: 1e-1 # Initial warm-up factor.
  cosine_epochs: 14 # Number of epoch of cosine decay.
  cosine_factor: 1e-2 # Final decay factor.
trainer:
  id: null # ID of the experiment for logging purposes, if set to null, it will be assigned automatically. 
  seed: 1 # Seed for reproducibility.
  max_batch_size_per_gpu: 8 # Maximal batch size for the GPU.
  precision: 32 # Numerical precision, can be 32, 16, bf16, etc...
```

### Tuning the Model
In the case where we are tuning our model manually, first, we aim to find a reasonable starting point in the hyperparameter space.

Given a `model.backbone`, we should the hyperparameters in the following order:
1. Find the right number of `optim.max_epochs`, until we clearly see that the model is over-fitting.
2. Adjust `optim.lr` and `optim.batch_size` to have a smoothly decreasing training loss and to minimise the discrepancy between the training and validation curves.
3. Tune the `scheduler` parameters to add a warm-up at the start of training, and to slowly settle in a local minimum with the cosine decay.

Once this reasonable starting point is found, we should explore other parameters to tune such as the `loss.label_smoothing`, to make the model less confident over its own predictions, or even experiment with another `model.backbone`.
This might require to explore again the parameters tuned above, but it will give a sound starting point, from which we can iterate.

We can also use automated or semi-automated methods to tune the hyperparameters, such as the multi-armed bandit (https://en.wikipedia.org/wiki/Multi-armed_bandit).
Many frameworks, such as `optuna` (https://optuna.org), can easily be integrated to the codebase to handle this.

### Analysing Results
The results of the preliminary run on my laptop are available at https://wandb.ai/guillaumerochette/aagi_assignment.
It contains the training and validation curves for each of the metrics, e.g. accuracy and intersection-over-union, for the semantic segmentation tasks.
We can find visualisations of the network predictions for both training (`256x256`) and validation (`2048x1024`) with captions depicting the performance of the model for that given image.

Even though the model parameters are not tuned, and this was just an initial guess, the model is able to recognise objects within images.
The training curves are a bit noisy, which might suggest that the learning rate is a bit high or the batch size is too small.
However, the validation curves are increasing smoothly over the epochs and are not plateauing yet, which indicates that training for longer should be beneficial.

As a criticism of the current results, I think it would be beneficial to first pre-train the segmentation model over larger scale datasets, such as COCO (https://cocodataset.org), which contains over 300K images, before fine-tuning on this dataset, which contains only 5K images.
Moreover, the limited input size of the backbone (`256x256`), given the current implementation, makes it impractical for large semantic segmentation maps.
At validation, the large maps of `2048x1024` are tiled into `256x256` chunks for inference, which makes the results not entirely smooth and hinders performance.

This dataset allows to train models to recognise a multitude of objects in urban environments.
However, the data was only captured from the front of a car, and this might be a hindrance to the aspect of being able to recognise target objects from a range of viewpoints.

This is however a data-dependent problem, by simply changing or aggregating multiple datasets, we could overcome this potential limitation.

This system is ready to detect and segment areas of texture, such as rust or invasive vegetation.
It only requires a dataset containing such data and their associated labels.

---
## State-of-the-art Awareness

### Transformer Block and Attention

A transformer block is composed of two sub-blocks:
- An attention module, either self-attention for an encoder or cross-attention for a decoder.
- A position-wise feed-forward network. 


#### Attention Module
The attention module aims at creating an operation similar to a dictionary look-up, in the sense that given a query, e.g. an unknown word, we are going to search a dictionary to find the most closely related key, e.g. a word defined by the dictionary, and read out its associated value, e.g. the definition of that word.

It differs from a traditional dictionary look-up on two aspects:
- It is a differentiable selection operation, which realises a convex combination, rather than a discrete selection.
- The key/value pairs are not fixed, but are rather context dependent.

We use a pythonic/pseudo-code notation, where `x : (M, N)`, defines `x` as an `M-by-N` matrix, and `a @ b` defines a matrix product between `a` and `b`.

It operates as follows, given,
- the queries, `q : (T, E)`,
- the key/value pairs, `k, v : (S, E)`.

We compute the affinity between the queries and keys, by performing an outer matrix product, e.g. a dot product between each column vector:
`p = q @ k.T : (T, S)`

This gives us what we call the attention matrix (or map), which for each entry of the matrix depicts the affinity between that query embedding and that key embedding.
It is essential to understand that a dot product measure the alignment or affinity between two vectors.
If it is positive, then the two vectors pointing in the same direction, if negative they are pointing in opposing directions, if null, then they are pointing in perpendicular directions.

This attention matrix is then transformed by a `softmax`, operation, which normalises the rows so that their sum equals 1,
`s = softmax(p) : (T, S)`

This gives us the coefficients of the convex combinations to select from the values,
`o = s @ v : (T, E)`

Going back to the discrete dictionary look-up example, this would be happening if the softmaxed dot product between the queries and keys was yielding a sparse matrix, e.g. one value on each row valued to 1 and the rest to 0, then we would simply pick the values for each queries.
However, in the machine learning case, the attention maps are rarely sparse, but instead enable to form convex combinations of multiple values.

##### Self-Attention and Cross-Attention

We refer to self-attention, an attention module where the queries, keys and values are computed as projection from the same input embeddings `x : (N, E)`,

`q = x @ w_q + b_q : (N, E)`

`k = x @ w_k + b_k : (N, E)`

`v = x @ w_v + b_v : (N, E)`

While we refer to cross-attention, when the queries originate from a target embedding `x : (T, E)` and the key/values pair from a source embedding `y : (S, E)`,

`q = x @ w_q + b_q : (T, E)`

`k = y @ w_k + b_k : (S, E)`

`v = y @ w_v + b_v : (S, E)`

#### Extracting Attention Maps in PyTorch

To extract attention maps, one simply need to do:
```python
mha = torch.nn.MultiheadAttention(...)
query = ...
key = ...
value = ...

output, attention_map = mha(query, key, value, average_attn_weights=True/False)
```
Depending on the value of `average_attn_weights`, the `attention_map` will/won't be averaged across heads.
In any case, that is the maps we are looking for.


#### Position-Wise Feed-Forward Network

This is simply two fully-connected linear layers, with a non-linear activation inbetween, which processes every embedding independently.


### Vulnerability to Noise and Adversarial Attacks

Current neural architectures are vulnerable to noise and adversarial attacks because they learn correlations between features, but do not possess causal reasoning abilities.
Therefore, it is possible to engineer data examples, which will cause the network to be misled by altering/corrupting partially or totally the example.

Networks can be trained to minimise their exposure to certain types of attacks, for example noises or patch attacks (https://arxiv.org/abs/2009.08194), but there is no one-fits-all solution to this.

The consequences of successful attacks are serious, as they will alter the downstream decision process with misleading information, which can pose serious threats to safety of humans.

### Hyperparameter Tuning

Depending on the number of hyperparameters, we can either just do:
- A manual tuning using `n`-ary searches, based on the assumption that the hyperparameter loss landscape is roughly convex.
- An automated tuning approach using a multi-armed bandit algorithm, this can be easily implemented using frameworks such as `optuna` or `sigopt`.

The first one is only viable when the number of hyperparameter is low, or if the compute resources are limited.

### Parallelism in Deep Learning
- Data Parallelism: It consists in distributing the compute by replicating the model and its parameters across multiple nodes, over a single machine with multiple GPUs or even multiple machines with multiple GPUs.
- Model Parallelism: It consists in splitting a large model into, usually sequential, chunks across multiple GPUs, e.g. layers 1 to 3 on the first GPU, 4 to 6 on the second GPU and so on. 
- Pipeline Parallelism: It extends the model parallelism by processing sub-batches of data sequentially on each chunk in order to reduce the idle time of the subsequent GPUs, which would be awaiting data otherwise.


### Generative Adversarial Network for Synthetic Data Generation

Generative adversarial networks are a powerful tool to generate new synthetic data.
The main reason behind this is that the concept of the adversarial game is to mimic a real data distribution.

The objective of the generator is to map from a classic distribution, often Gaussian, to any data distribution.
Meanwhile, the objective of the discriminator is, when presented two data examples, to be able to distinguish which one is drawn from the real data distribution, e.g. the dataset, and the one that is drawn from the fake distribution, e.g. created by the generator.
Having derived this adversarial objective we are now able to train simultaneously the two competing networks.

Once the min-max game stabilises, and the discriminator is not capable of distinguishing the real from the fake data distribution, then our generator network is now capable of mapping from a classic distribution to a distribution closely related to the real data distribution.
We are now able to use the generator to produce synthetic data for that context.

GANs are often a good alternative to generate synthetic data nowadays, instead of solely relying on hand-crafted parametric model, because the requirements in terms of adaptability, caused by the adversarial nature, force the generator to be more diverse and realistic than human-made parametric models.
However, there is a drawback, the adversarial game is unstable by nature, which makes them tricky to train, and often requires in-depth exploration of the hyperparameter space and long training times. 
