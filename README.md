# UMRL--using-Cycle-Spinning
Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining


## To test UMRL:
python umrl_test.py --dataroot ./facades/validation --valDataroot ./facades/validation --netG ./pre_trained/Net_DIDMDN.pth

## To train UMRL:
python umrl_train.py  --dataroot <dataset_path>  --valDataroot ./facades/validation --exp ./check --netG ./pre_trained/Net_DIDMDN.pth

## To test UMRL using Cycle Spining:
python umrl_cycspn_test.py --dataroot ./facades/validation --valDataroot ./facades/validation --netG ./pre_trained/Net_DIDMDN.pth

## To train UMRL using Cycle Spining:
python umrl_cycspn_train.py  --dataroot <dataset_path>  --valDataroot ./facades/validation --exp ./check --netG ./pre_trained/Net_DIDMDN.pth

## Acknowledgments
Thanks for the help from [He Zhang](https://sites.google.com/site/hezhangsprinter/)
