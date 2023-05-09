Need to setup:
- Trac-IK: `sudo apt-get install ros-noetic-trac-ik`
- RealSense (librealsense)
- ar_track_alvar
- [yumi](https://github.com/zhyma/yumi/tree/coil_dev)(fork from [kth-ros-pkg](https://github.com/kth-ros-pkg/yumi/). Please use `bimanual_rws` branch)
- [ariadne_plus](https://github.com/zhyma/ariadne_plus/tree/coil_dev)(fork from [lar-unibo/ariadne_plus](https://github.com/lar-unibo/ariadne_plus), using the [same trained model](https://drive.google.com/file/d/1rwyuUeltodsZjm53q6_46a8T-dRh1pnw/view?usp=sharing). Please select `coil_dev` branch, with `srv` file modified)
	- `pip3 install arrow termcolor igraph scikit-image pytorch-lightning==1.7.1 torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu`

To run the package
  - Use `roslaunch rs2pcl demo.launch`
  - Run `python config.py` to get the rod and the rope's estimation
  - Run `python winding.py` to start wrapping process.

Saved file format:
  - Rod's information is saved to `rod_info.pickle`.
  - Rope's information is saved to `rope_info.pickle`.
  - `save/log.txt`: advance, R, L', extra notes
  - `save/param.txt`: 
    - line1: adv_s, r_s, lp_s. "s" stands for "stable parameters that works."
    - line2: adv_n, r_n, lp_n. "n" stands for "next parameters that shoulb be tested."
    - line3: last_adv_fb, last_len_fb. As the name suggested.
    - line4: [r is stable?, adv is stable?]