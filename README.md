# PolarPoint-BEV
This is the official pytorch implementation of PolarPoint-BEV.

The codes and dataset will be released when the paper is published.

## Setup
Download and setup CARLA 0.9.10.1 (from TCP https://github.com/OpenDriveLab/TCP)
```
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..
```

Clone this repo and build the environment

```
git clone https://github.com/lab-sun/Polar-BEV.git
cd Polar-BEV
conda env create -f environment.yml --name Polar-BEV
conda activate Polar-BEV
```


## Dataset
Download the datasets and then extract it in the file of `Data`

The Control Prediction Module of the XPlan network is firstly pre-trained on the dataset from [TCP](https://github.com/OpenDriveLab/TCP)

To train the XPlan network, please refers to: http://

## Pretrained weights：
* Download the pretrained weights and then extract it in the file of `weight`
* The link for pretrained weights is: https: //

## Evaluation
The evaluation is performed in the Carla Simulator.

Step1: Launch the Carla server,
```
cd CARLA_ROOT
./CarlaUE4.sh --world-port=2000 -opengl
```
Set the parameters in the ``leaderboard/scripts/run_evaluation.sh``.

Step2: Start the evaluation

```
sh leaderboard/scripts/run_evaluation.sh
```



## Citation
If you found this code or dataset are useful in your research, please consider citing
```
...
```
