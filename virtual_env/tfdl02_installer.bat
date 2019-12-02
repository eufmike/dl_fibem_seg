@ECHO OFF
CALL conda.bat create -n tfdl02 python=3.6 -y
CALL conda.bat activate tfdl02
CALL conda.bat install -c conda-forge numpy scipy matplotlib imutils pandas tqdm -y
CALL conda.bat install -c conda-forge jupyter -y
CALL conda.bat install -c conda-forge pillow scikit-image scikit-learn -y
CALL conda.bat install -c anaconda tensorflow-gpu -y
pip install opencv-python==3.4.2.16
