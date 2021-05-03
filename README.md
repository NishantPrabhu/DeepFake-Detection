# TechSoc IHC Deep Learning Hack 2021: Track 1 - DeepFake Detection
**Team**: Nishant Prabhu, Shrisudhan G.

Our solution for DL Hack 2021 Track 1 by Analytics Club, IITM. Network to detect images manipulated using deep learning.

# Trained model
ResNet18 with modified initial layers: [model](https://drive.google.com/file/d/1hn1E8O-2lOxy4J7in7eRCDexiwT3zM3o/view?usp=sharing) (**Leaderboard score**: 0.12088)

# Usage
The code to train the best model was written by Shrisudhan G. To re-train the model, run all the cells in `resnet.ipynb`. To perform inference on the test set by using the trained model, run the following command inside `main/`:

```
python3 main.py --config 'configs/main.yaml' --task 'custom_test'
```

Ensure that `train` and `test` folders as provided in the contest are available in a directory with name `track1_data` on the same directory level as `main` before you run the above command.
