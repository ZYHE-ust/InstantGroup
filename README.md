# InstantGroup: Instant Template Generation for Scalable Group of Brain MRI Registration

This is the official Pytorch implementaiton of ["InstantGroup: Instant Template Generation for Scalable Group of Brain MRI Registration."](https://ieeexplore.ieee.org/document/11080188) IEEE Transactions on Image Processing (2025) by Ziyi He and Albert C. S. Chung.

## Prerequistes
- `Python 3.6.13+`
- `torch 1.10.2`
- `numpy`
- `glob`
- `tqdm`

## Training
`python train_instantGroup.py`

Please implement your own data loading function `load_img` in `utils.py` and pretrain the registration model by `python train_reg.py` for stable and efficient training of InstantGroup model.

## Testing
`python test_instantGroup.py`

## Citations
    @article{he2025instantgroup,
      title={InstantGroup: Instant Template Generation for Scalable Group of Brain MRI Registration},
      author={He, Ziyi and Chung, Albert CS},
      journal={IEEE Transactions on Image Processing},
      year={2025},
      publisher={IEEE}
      }

## Acknowledgements
Some code of the project is modified from repository [voxelmorph](https://github.com/voxelmorph/voxelmorph).
