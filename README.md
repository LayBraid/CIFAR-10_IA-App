# CIFAR-10_IA-App :rocket:

This is an application to recognize images from the CIFAR-10 dataset

A Gradio integration is present. You can launch the app with the argument "1" to draw and get a real time result. Or you can launch the app with argument "2" to drag and drop images and get a result.

The app will first download the dataset if not already done.

After, if this is the first time you run the app on your machine, the program will train the model.

# Usage

## To draw an image

```
./dataset_loader.py 1
```

## To drag and drop an image

```
./dataset_loader.py 2
```

# Developer

| [<img src="https://github.com/LayBraid.png?size=85" width=85><br><sub>Clément Loeuillet</sub>](https://github.com/LayBraid)
| :---: |