# Google Coral Accelerator Demo  
![image](https://github.com/allen050883/Deeplearning/blob/master/google_coral_demo/google_coral_resize.jpg)  
#### Support System
Connects via USB to any system running Debian Linux (including Raspberry Pi), macOS, or Windows 10.  
#### Support Tensorflow Lite  
No need to build models from the ground up. TensorFlow Lite models can be compiled to run on the Edge TPU.  
#### Notice  
Google coral Accelerator needs to use USB 3.0.  
  
  
## Device and OS  
Rasberry Pi 4  
Raspberry Pi OS (32-bit) with desktop (https://www.raspberrypi.org/downloads/raspberry-pi-os/)  
  
## Getting started with Google Coral Accelerator  
#### Install edgetpu package and reboot  
```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
sudo apt-get install python3-edgetpu
sudo reboot now
```
#### Setting virtual environment  
```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py
sudo rm -rf ~/.cache/pip
sudo pip3 install virtualenv virtualenvwrapper
nano ~/.bashrc
```
#### Add lines from bottom of this file
```
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```
#### Continue  
```
source ~/.bashrc
mkvirtualenv coral -p python3
```
If get ERROR: Environment '/home/pi/.virtualenvs/cv' does not contain an activate script, please change the version:  
```
sudo pip3 install virtualenv virtualenvwrapper=='4.8.4'
```
#### Show the edgetpu package place, and see the Line 7
```
dpkg -L python3-edgetpu
#In the root dir: /usr/lib/python3/dist-packages/edgetpu
```
#### Create a sym-link to that path from our virtual environment site-packages  
```
cd ~/.virtualenvs/coral/lib/python3.7/site-packages
ln -s /usr/lib/python3/dist-packages/edgetpu/ edgetpu
cd ~
```
#### Check the virtual environment and edgetpu version  
```
workon coral
$ python
>>> import edgetpu
>>> edgetpu.__version__
'2.12.2'
```
If get error, please install numpy first: pip install -U numpy  
#### Install packages
```
workon coral #check the virtual environment
pip3 install "picamera[array]" # Raspberry Pi only
pip3 install numpy
pip3 install opencv-contrib-python==4.1.0.25
pip3 install imutils
pip3 install scikit-image
pip3 install pillow
```
#### Install edgetpu example and chmod
```
sudo apt-get install edgetpu-examples
sudo chmod a+w /usr/share/edgetpu/examples
```
#### check the all files  
```
workon coral
cd /usr/share/edgetpu/examples
tree --dirsfirst
```
#### try some examples  
```
python3 classify_image.py \
	--mode models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
	--label models/inat_bird_labels.txt \
	--image images/parrot.jpg 
```
Reference:  
https://www.pyimagesearch.com/2019/04/22/getting-started-with-google-corals-tpu-usb-accelerator/  
https://raspberrypi.stackexchange.com/questions/108740/error-environment-home-pi-virtualenvs-cv-does-not-contain-an-activate-scrip 
https://stackoverflow.com/questions/20518632/importerror-numpy-core-multiarray-failed-to-import  
