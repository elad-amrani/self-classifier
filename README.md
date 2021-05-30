# Self-Classifier: Self-Supervised Classification Network

Implementation of the paper "Self-Supervised Classification Network". Self-Classifier is a self-supervised end-to-end classification neural network. It learns labels and representations simultaneously in a single-stage end-to-end manner.

## Setup

1. Install Conda environment:

        conda env create -f ./environment.yml

2. Install Apex with CUDA extension:
 
        export TORCH_CUDA_ARCH_LIST="7.0"  # see https://en.wikipedia.org/wiki/CUDA#GPUs_supported
        pip install git+git://github.com/NVIDIA/apex.git@4a1aa97e31ca87514e17c3cd3bbc03f4204579d0 --install-option="--cuda_ext"         


##Citation

If you find this repository useful in your research, please cite:

@article{amrani2021self,
  title={Self-Supervised Classification Network},
  author={Amrani, Elad and Bronstein, Alex},
  journal={arXiv preprint arXiv:2103.10994},
  year={2021}
}

