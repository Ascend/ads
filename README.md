## Description

+ This repo provides some speed-up abilites for AI-models in autonomous driving system, which can help these models running fatser on ascend device.

+ The main three parts is as follows:
    + **common**, general modules, which contain some fused pytorch modules of which provide speed-up ability for models runnning on ascend device. 
    + **motion**, the motion prediction modules, which are used for trace-planning and motion-prediction. The modules contain some pytorch-custom-apis, the kernels of which are affinitive on ascend device.
    + **perception**, the perception modules, which are used for 3D detection and segmentation. It helps recognize the components on the road.