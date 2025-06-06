# DeepGF: Glaucoma Forecast Using the Sequential Fundus Images
- This is the official repository of the paper "DeepGF: Glaucoma Forecast Using the Sequential Fundus Images" from **MICCAI 2020**[[Paper Link]](https://link.springer.com/chapter/10.1007/978-3-030-59722-1_60, "Paper Link")[[PDF Link]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-59722-1_60.pdf)

![framework](./imgs/fullnet_101.jpg)

## 1. Environment
- Python >= 3.5
- Tensorflow >= 1.4 is recommended
- opencv-python
- sklearn
- matplotlib


## 2. Dataset

*** Update ***

**Of note, after undergoing a period of open access followed by a more stringent ethics review, the SIGF database is now available again upon request (to xw405@cam.ac.uk).**

*** Update ***

**SIGF-database is currently undergoing ethics review and not available now.**

** Update ***

1. The training data and testing data is from the [[SIGF-database]](https://www.dropbox.com/s/a0p05573xx37lfx/SIGF-database.rar?dl=0, "Official SIGF"). Contact [liu.li20@imperial.ac.uk] or [xfwang@buaa.edu.cn] for password of the shared data in dropbox. Below is an example of our SIGF database. 

![Database](./imgs/database.jpg)

2. Put the training and test images and the labels in the directory:
```
'./data/train(test)/image(label)/all/'
```

3. Obtain the polar and attention data from the  [[Dropbox]](https://www.dropbox.com/s/q23i1le5vhs9ilv/Polar-Attention.zip?dl=0, "Attention and Polar"). Below is an example of the polar and attention map of a glaucoma fundus image.

![Polar-Attention](imgs/fundusimage.jpg)


4. Put the attention and polar images in the directory:
```
'./data/'
```

## 3. Training
The details of the hyper-parameters are all listed in the `train.py`. Use the below command to train our model on the SIGF database.

```
    python ./train.py 
```

## 4. Test
Download the pre-trained model in [[Dropbox]](https://www.dropbox.com/s/e1oebawbp5wlpvm/pretrained_model.zip?dl=0). Then put the file in tghe directory of 
`pretrained_model`. Use the below command to test the model on the SIGF database.
```
    python ./test.py 
```

## 5. Compared Methods

The network re-implenmentation of [[Chen et al.]](https://ieeexplore.ieee.org/abstract/document/7318462/, "Chen") is in the file of:
`chen_net.py`
and from the directory of `./Compared Methods`




## 6. Ablation Study

If you are interested in our ablation study, please see `./Ablation study`




## 7. Network Interpretability

1. If you are interested in the visualization method and results used for showing the interpretability 
of our method, please refer to the directory of `./saliency`



2. Or you can just see the images in the directory of `./visualization_result`
for more visualization results. Some examples of the visualization rsults are shown here.

![Database](./imgs/figure1.jpg)


## 8. Citation
If you find our work useful in your research or publication, please cite our work:
```
@article{Li2020deep,
  title={DeepGF: Glaucoma Forecast Using the Sequential Fundus Images.},
  author={Li, Liu and Wang, Xiaofei and  Xu, Mai and Liu, Hanruo},
  journal={MICCAI},
  year={2020}
}
```

## 9. Contact
If any question, please contact [xfwang@buaa.edu.cn]

## 10. Supplementary Materials

[[Supplementary]](https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-59722-1_60/MediaObjects/505218_1_En_60_MOESM1_ESM.pdf)
