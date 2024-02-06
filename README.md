# BOWLL: A Deceptively Simple Open World Lifelong Learner
This package contains the code for BOWLL: A deceptively simple Open World Lifelong Learner. 

Abstract: The quest to improve scalar performance numbers on predetermined benchmarks
seems to be deeply engraved in deep learning. However, the real world is seldom
carefully curated and applications are seldom limited to excelling on test sets. A
practical system is generally required to recognize novel concepts, refrain from
actively including uninformative data, and retain previously acquired knowledge
throughout its lifetime. Despite these key elements being rigorously researched
individually, the study of their conjunction, open world lifelong learning, is only a
recent trend. To accelerate this multifaceted field’s exploration, we introduce its
first monolithic and much-needed baseline. Leveraging the ubiquitous use of batch
normalization across deep neural networks, we propose a deceptively simple yet
highly effective way to repurpose standard models for open world lifelong learning.
Through extensive empirical evaluation, we highlight why our approach should
serve as a future standard for models that are able to effectively maintain their
knowledge, selectively focus on informative data, and accelerate future learning.

## Credits
* We implemented DeepInversion using the official repository [DeepInversion](https://github.com/NVlabs/DeepInversion)
* We adapted Active Query pool mechanism from [DDU](https://github.com/omegafragger/DDU)

## Environmental Setup

You can install packages from `requirements.txt` after creating your own environment with `python 3.7.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Please download the DeepInversion package from the github repository and place it in the root directory.
This generates pseudo-images for BOWLL.

## Run Experiments
You can reproduce the experiments in the paper by running the following command:

```bash
python ../bowll_mnist.py \
		--training_batch_size 256 \
        --acquisition_batch_size 256 \
		--test_batch_size 256 \
		--ood_batch_size 4 \
	 	--n_domains 4 \
		--n_epochs 1 \
		--n_repeats 5 \
		--buffer_size 5000 \
        --arch alexnet \
        --path_to_weights ...


python ../bowll_cifar10.py \
		--training_batch_size 256 \
        --acquisition_batch_size 256 \
		--test_batch_size 256 \
		--ood_batch_size 8 \
	 	--n_timestpes 5 \
		--n_epochs 3 \
		--n_repeats 5 \
		--buffer_size 5000 \
        --arch resnet18 \
        --path_to_weights ... \

```

