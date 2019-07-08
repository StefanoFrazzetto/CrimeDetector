# Crime Detector

This project has been created as part of my undergraduate degree dissertation, 
which is available [here](https://stefanofrazzetto.com/dissertation/).

The software allows to combine feature extraction techniques and machine learning algorithms to analyse different 
datasets. Several metrics can be plotted in order to determine which process produces the best results. 

The following Jupyter notebooks show how simple the process is:

- [Formspring](./Formspring.ipynb)
- [PAN-12](./PAN12.ipynb)

## Abstract

In recent years, online social networks have experienced a massive growth in their user base. Crime Survey for England and Wales on fraud and computer misuse reports an estimated number of 83,125 online offences from 2017 to 2018, including, but not limited to, harrassment and stalking, obscene publications, and child sexual offences. Machine learning algorithms can be employed to detect such crimes in social networks, online games, and generally in contexts involving online exchanges of messages. The objective of this project was to compare different feature extraction techniques and machine learning algorithms on two datasets, namely PAN-12 and Formspring v4, respectively containing instances of predatory messages and cyber bullying traces. In order to determine which approach was most effective in detecting and classifying these crimes, a software was created to test the different scenarios on each of the datasets. This allowed to obtain different metrics and to compare them across all the proposed scenarios. The final results proved to be satisfactory, with the highest values for Matthews’ Correlation Coefficient peaking at 0.74 for the PAN-12 dataset, and at 0.64 for the Formspring dataset. However, the quality of the results obtained remains inherently linked to the quality of the available datasets. The implementation of such algorithms in online environments would help provide a safer experience for both children and adults alike.

## License

Copyright 2019 Stefano Frazzetto

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.