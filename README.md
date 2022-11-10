# ProGAN (Progressive Generative Adversarial Network) trained on various image datasets

The purpose of this project is to expand knowledge of traditional GAN architectures by exploring NVIDIA's Progressive GAN research. [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://research.nvidia.com/publication/2018-04_progressive-growing-gans-improved-quality-stability-and-variation)

### Install Dependencies
* `pip install -r requirements.txt` - from project root

### Initialize the Project
* `python utils.py init` - download all logs for exploration then init empty models and results directory
* You must **REMOVE** the logs folder this command will **NOT** overwrite an log files.

### Download the Pre-trained models and results
* `python utils.py download` - download the pre-trained model files and image sets.
* Options = `[cars, cyber, dogs, faces, potatoes]`; default = `faces`
* This command will NOT override existing models you must delete both `imgs` and `models` folders to download a new image and model set

### Explore Tensorboard Results
* `tensorboard --logdir logs` - to start the tensorboard and explore the current models training progress.

### Remove Duplicate Images
* `python utils.py removedups` - this will remove duplicate images using an image hash function.
* The `hash_size` for this function can be customized to have a looser hash criteria i.e accepting a larger difference between "duplicates"
* This function will output how many duplicates have been removed and a path list for all duplicate images. **NOTE:** paths are not paired with duplicates.

### Train the model
* **WARNING! This can overwrite existing models!** 
* `python train.py` - continue training or re-train model.
* Changing the `LOAD_MODEL` global in `config.py`; chooses between re-training (`False`) or continuing training (`True`) for the model.

### Generate Samples
* `python utils.py sample` - generates sample images, by default this will generate 10 images at 64x64.
* This command can be customized to generate a chosen amount of images at a chosen size `python utils.py sample <num_images> <size_factor>`
* Size factors: `0 = 4x4, 1 = 8x8, 2 = 16x16, 3 = 32x32, 4 = 64x64, 5 = 128x128, 6 = 256x256, 7 = 512x512, 8 = 1024x1024`

### Preview Image Transforms
* `python utils.py transform` - generates output images, by default this will output all batches at 512x512.
* This command can be customized to generate a chosen amount of batches `python utils.py transform <num_batches>`

### References

```
@misc{https://doi.org/10.48550/arxiv.1701.07875,
  doi = {10.48550/ARXIV.1701.07875},
  url = {https://arxiv.org/abs/1701.07875},
  author = {Arjovsky, Martin and Chintala, Soumith and Bottou, Léon},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Wasserstein GAN},
  publisher = {arXiv},
  year = {2017},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
```
@misc{https://doi.org/10.48550/arxiv.1704.00028,
  doi = {10.48550/ARXIV.1704.00028},
  url = {https://arxiv.org/abs/1704.00028},
  author = {Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Improved Training of Wasserstein GANs},
  publisher = {arXiv},
  year = {2017},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
```
@misc{https://doi.org/10.48550/arxiv.1710.10196,
  doi = {10.48550/ARXIV.1710.10196},
  url = {https://arxiv.org/abs/1710.10196},
  author = {Karras, Tero and Aila, Timo and Laine, Samuli and Lehtinen, Jaakko},
  keywords = {Neural and Evolutionary Computing (cs.NE), Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Progressive Growing of GANs for Improved Quality, Stability, and Variation},
  publisher = {arXiv},
  year = {2017},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```