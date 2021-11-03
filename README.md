# Bird_classification

This repository is the implementation of SSL Face classification

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training(SSL)

To train the SimSiam model(s), run this command:

```train_SimCLR
python trainSimCLR.py --data_path <train_data_path>  --batch_size 8 --lr 2e-4 --output_foloder <model output path> --epochs 100
```

```train_SimSiam
python trainSimSiam.py --data_path <train_data_path>  --batch_size 8 --lr 2e-4 --output_foloder <model output path> --epochs 100
```

To train the VICReg model(s), run this command:

```train_VICReg
python trainVICReg.py --data_path <train_data_path>  --batch_size 8 --lr 2e-4 --output_foloder <model output path> --epochs 100
```

To train the train_SimSiam + VICReg model(s), run this command:

```trainSimSiamAndVICReg
python trainSimSiamAndVICReg.py --data_path <train_data_path>  --batch_size 8 --lr 2e-4 --output_foloder <model output path> --epochs 100
```

* scheduler use `CosineAnnealingLR`
* optimizer  use `SGD`, momentum = 0.9, weight_decay = 1e-4
* Data augmentation(RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, GaussianBlur)

## Training(Facenet)

```train_facenet
python trainSimSiam.py --data_path <train_data_path> --batch_size 8 --lr 2e-4 --pretrain_model_path <pretrain_model_path> --output_foloder <model output path> --epochs 100
```

* scheduler use `CosineAnnealingLR`
* optimizer  use `SGD`, momentum = 0.9, weight_decay = 1e-4
* Data augmentation(RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, GaussianBlur)

## Evaluation

To evaluate my model, run:

```eval
python eval.py --data_path <eval_data_path> --classes_path <classes.txt> --training_labels_path <training_labels.txt> --model_path <model_path>
```

## Reproduceing Submission

[model link](https://drive.google.com/file/d/1FXQF4Pbpco3FkNbiQxSbN__ZSSd-ypsV/view?usp=sharing)

```inference
python inference.py --data_path <test_image_path> --classes_path <classes.txt> --test_filename_path <testing_img_order.txt> --model_path <model_path>
```
>📋 Will output `answer.txt`
## Pre-trained Models

You can download pretrained models here:

- [ImageNet-21K Pretraining for the Masses
](https://github.com/Alibaba-MIIL/ImageNet21K) trained on ImageNet.

- Or use timm to load the model

```
model = timm.create_model('vit_base_patch16_224_miil_in21k', pretrained=True)
```

## Results

Our model achieves the following performance on :

### Image Classification

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My best model      |     62.17%      |     52.45%     |
