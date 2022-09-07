Need to setup:
- Trac-IK: `sudo apt-get install ros-noetic-trac-ik`
- RealSense (librealsense)
- ar_track_alvar
- [yumi](https://github.com/zhyma/yumi/tree/coil_dev)(fork from kth-ros-pkg, `coil_dev` branch)
- [ariadne_plus](https://github.com/zhyma/ariadne_plus/tree/coil_dev)(fork from lar-unibo/ariadne_plus, `coil_dev` branch, with `srv` file modified)
	- `pip3 install arrow termcolor igraph scikit-image pytorch-lightning==1.7.1 torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu`

To run the package, use `roslaunch rs2pcl demo.launch` first, then `python main.py`