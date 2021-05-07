# Fully adaptive radar 
![](https://img.shields.io/badge/python-v3.7-blue) ![](https://shields.io/badge/license-Apache-blue) ![](https://img.shields.io/badge/Cognative-Radar-brightgreen)

This repo contains a reproduction of the method and results presented in 

    Bell, Kristine L., et al. "Fully adaptive radar for target classification." 2019 IEEE Radar Conference (RadarConf). IEEE, 2019.

The method is related to rescource management in "cognative radar" and comprise:

- A categorical random variable is estimated recursively from covariates drawn from sensor measurements. 
- If the entropy of the random variable is low then no measurement is taken - hopefully enabling the radar to perform some other useful task
- The entropy of the categorical random variable increases over time in accordance with a transition model.

![](images/visualization.jpg)