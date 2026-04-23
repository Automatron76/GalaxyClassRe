# References

All external sources used in this project.

## Dataset

- **Galaxy Zoo 2 Catalog**
   gz2_hart16.csv.gz
  https://data.galaxyzoo.org/

- **Galaxy Zoo 2 Paper**
  Willett et al. 2013, MNRAS, 435, 2835
  https://academic.oup.com/mnras/article/435/4/2835/1022913

- **SDSS SkyServer Image Cutout API**
  Used to download 256x256 JPEG galaxy images via RA/Dec coordinates
  https://skyserver.sdss.org/dr14/en/help/docs/api.aspx

## Model & Architecture

- **ResNet18 Pretrained Model**
  Torchvision pretrained on ImageNet (IMAGENET1K_V1 weights)
  https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

- **ImageNet Normalisation Values**
  Official mean and std values required for all torchvision pretrained models
  https://docs.pytorch.org/vision/stable/models.html

## PyTorch Documentation

- **Dataset & DataLoader Pattern**
  https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

- **AdamW Optimizer**
  https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

- **CrossEntropyLoss**
  https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 