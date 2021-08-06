
511 CCTV Camera Vehicle Detection Using DAIU Net and STMAP For Advanced Traveler Information System (ATIS)
-----
Scanline MethodS for [NJ511](https://511nj.org/camera) Traffic Camera Network

<p align="center"><img src="https://github.com/TeRyZh/Detection-is-Tracking-511-CCTV-Camera-Vehicle-Detection-Using-STMap-and-DAIU-Net/blob/main/Figures/selected_testing_sites.png" /></p>

Highlights
----------
* Proposed Dual Attention Inception Neural Network for STmap Detection that has bettern performance than State-of-the-art image segmentation Models
* An Efficient STMap-based Vehicle Detection Model Under Various Illuminations, Infrastructure Noises, Weather Conditions

Abstract
--------
This paper developed a Spatial-temporal Map (STMap) based vehicle detection method for 511 camera networks as an add-on toolbox for the traveler information system platform. The U-Shaped Dual Attention Inception (DAIU) deep learning model was designed, given the similarities between the STMap vehicle detection task and the medical image segmentation task. The inception backbone takes full advantage of diverse sizes of filers and the flexible residual learning design. The channel attention module augmented the feature extraction for the bottom layer of the UNet. The modified gated attention scheme replaced the skip-connection of the original UNet to help the model to reduce irrelevant features learned from earlier encoder layers. The designed model was tested on NJ511 traffic video for different scenarios covering rainy, snowy, low-illumination, and signalized intersections from a key, strategic arterial in New Jersey. Based on segmentation model evaluation metrics, the DAIU net has shown better performance than other mainstream neural networks. The DAIU based STMap vehicle detection is also compared against the state-of-the-art solution for infrastructure-based traffic video understanding and demonstrates superior capability. The code for the proposed DAIU model and reference models are made public, and the labeled STMap data to facilitate future research.

## Vehicle Detection Results

### [US1 at Menlo Park Dr Light Rain](https://www.youtube.com/watch?v=xzrfBH-zZOA&list=PLC4d9Yu1vCsl02xe5gP3HNMD38QLYpFCX&index=1)

### [US1 at Ryders Lane Snow](https://www.youtube.com/watch?v=wCv2EuXUoRA&list=PLC4d9Yu1vCsl02xe5gP3HNMD38QLYpFCX&index=2)

### [US1 MenloPark Dr Heavy Rain](https://www.youtube.com/watch?v=y6us4C5BQOs&list=PLC4d9Yu1vCsl02xe5gP3HNMD38QLYpFCX&index=3)

### [US1 Alexander Rd](https://www.youtube.com/watch?v=sWeNFbOVwF4)

License
-------
The source code is available only for academic/research purposes (non-commercial).


## Contributing
If you found any issues in our model or new dataset please contact: terry.tianya.zhang@gmail.com
