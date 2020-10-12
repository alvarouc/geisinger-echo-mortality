# Echo mortality model applied to Stanford Dataset
Demo of Geisinger's trained models applied to Stanford Ouyangs Dataset


1. Obtain Stanford Dataset

    Register at: https://echonet.github.io/dynamic/index.html#dataset

    Once given access, download and unzip the file EchoNet-Dynamic.zip

    In this demo, we unzipped the file to the folder path : /data/EchoData/aeulloacerna/EchoNet-Dynamic/

    Update the path to the data in the Makefile by modifying the variable `STANFORD_DATA` accordingly. 

2. Building container

    At the root of the repository type `make build`

    This will prepare a Docker image named geisinger-echo-mortality:v1. This image is used in the next steps to load and apply the models to the Stanford dataset.

3. Apply the AP4 model to Stanford Dataset

    At the root of the repository type `make run`

    This will open a docker container where the AP4 model will run 

    Two files will be generated: 
    - FileList_predicted.csv
    - stanford_risk_vs_ef.pdf

    