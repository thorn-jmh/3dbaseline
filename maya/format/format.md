3d-base-line输出格式，按帧储存,坐标以[0]节点为原点

```json
{
    "version": "3d-base-line",
    "pose_keypoints_3d": [x0,y0,z0,...,x16,y16,z16] //目前只支持单人
}
```

节点解释：

```python
JOINT_PONITS[0]  = 'Hip'       #盆骨
JOINT_PONITS[1]  = 'RHip'      #右股
JOINT_PONITS[2]  = 'RKnee'     #右膝
JOINT_PONITS[3]  = 'RFoot'     #右足
JOINT_PONITS[4]  = 'LHip'      #左股
JOINT_PONITS[5]  = 'LKnee'     #左膝
JOINT_PONITS[6]  = 'LFoot'     #左足
JOINT_PONITS[7] = 'Spine'      #脊柱中心
JOINT_PONITS[8] = 'Thorax'     #前胸
JOINT_PONITS[9] = 'Neck/Nose'  #颈部/鼻部
JOINT_PONITS[10] = 'Head'      #额头
JOINT_PONITS[11] = 'LShoulder' #左肩
JOINT_PONITS[12] = 'LElbow'    #左肘
JOINT_PONITS[13] = 'LWrist'    #左腕
JOINT_PONITS[14] = 'RShoulder' #右肩
JOINT_PONITS[15] = 'RElbow'    #右肘
JOINT_PONITS[16] = 'RWrist'    #右腕
```

![jpoints](https://s2.loli.net/2022/04/28/7aqdEvcMgFmz4Jk.png)