# UMRL--using-Cycle-Spinning
Uncertainty Guided Multi-Scale Residual Learning-using a Cycle Spinning CNN for Single Image De-Raining
[Rajeev Yasarla](https://scholar.google.com/citations?user=R8dwrxEAAAAJ&hl=en&oi=ao), [Vishal M. Patel](http://www.rci.rutgers.edu/~vmp93/)

[Paper Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yasarla_Uncertainty_Guided_Multi-Scale_Residual_Learning-Using_a_Cycle_Spinning_CNN_for_CVPR_2019_paper.pdf) (CVPR'19)

We present a novel Uncertainty guided Multi-scale Residual Learning (UMRL) network to address the single image de-raining. The proposed network attempts to  address this issue by learning the rain content at different scales and using them to estimate the final de-rained output.  In addition, we introduce a technique which guides the network to learn the network weights based on the confidence measure about the estimate.  Furthermore, we introduce a new training and testing procedure based on the notion of cycle spinning to improve the final de-raining performance.

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
