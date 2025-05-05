## This is the COMP3065 cv courese work of Kai Yang 20411990

## Prepare 
1 Create a virtual environment with Python >=3.8  
~~~
conda create -n py38 python=3.8    
conda activate py38   
~~~

2 Install pytorch >= 1.6.0, torchvision >= 0.7.0.
~~~
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
~~~

## Run
~~~
# on video file
python main.py --input_path input.mp4
or
python app.py #using GUI

~~~



I can't upload .pt files (larger than 360mb) due to github file size limit，please use this to download the .pt file, and put it in the **weights**floder.
https://drive.google.com/drive/folders/1rA6prLERdgrEViqdbOH1vDAJfdZ7qFaJ?usp=sharing

