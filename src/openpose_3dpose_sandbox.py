import pdb
from random import expovariate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow.compat.v1 as tf
import data_utils
import viz
import re
import cameras
import json
import os
import time
from predict_3dpose import create_model
import cv2
import imageio
import logging
import scipy as sp
from pprint import pprint
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

FLAGS = tf.app.flags.FLAGS

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

outf= 'maya/outputData.json' 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_anim_curves(anim_dict, _plt):
    # pdb.set_trace()
    val = np.array(list(anim_dict.values()))
    #读取所有帧的某个节点的 x,y 坐标
    for o in range(0,36,2):
        x = val[:,o]
        y = val[:,o+1]
        #红色虚线
        _plt.plot(x, 'r--', linewidth=0.2)
        #绿色
        _plt.plot(y, 'g', linewidth=0.2)
    return _plt

#读入2d_json文件
def read_openpose_json(smooth=True, *args):
    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    logger.info("start reading json files")
    #load json files
    json_files = os.listdir(openpose_output_dir)
    json_files = [file_name for file_name in json_files if file_name.endswith(".json")]
    # check for other file types
    #改成了sort
    json_files.sort(key=lambda x: int(x[:-5]))
    # pdb.set_trace()
    #cache dict
    cache = {}
    #
    smoothed = {}
    ### extract x,y and ignore confidence score
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(openpose_output_dir, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        data = json.load(open(_file))
        #take first person
        # pdb.set_trace()
        _data = data["people"][0]["pose_keypoints_2d"] if "pose_keypoints_2d" in data["people"][0] else data["people"][0]["pose_keypoints"]
        xy = []
        if len(_data)>=53:
            #openpose incl. confidence score
            #ignore confidence score
            for o in range(0,len(_data),3):
                xy.append(_data[o])
                xy.append(_data[o+1])
        else:
            #tf-pose-estimation
            xy = _data
        # pdb.set_trace()
        # get frame index from openpose 12 padding
        frame_indx = re.findall("(\d+)", file_name)
        logger.debug("found {0} for frame {1}".format(xy, str(int(frame_indx[-1]))))

        #body_25 support, convert body_25 output format to coco
        if len(_data)>54:
            # pdb.set_trace()
            _xy = xy[0:19*2]
            for x in range(len(xy)):
                #del jnt 8
                if x==8*2:
                    del _xy[x]
                if x==8*2+1:
                    del _xy[x]
                #map jnt 9 to 8
                if x==9*2:
                    _xy[16] = xy[x]
                    _xy[17] = xy[x+1]
                #map jnt 10 to 9
                if x==10*2:
                    _xy[18] = xy[x]
                    _xy[19] = xy[x+1]         
                #map jnt 11 to 10
                if x==11*2:
                    _xy[20] = xy[x]
                    _xy[21] = xy[x+1]
                #map jnt 12 to 11
                if x==12*2:
                    _xy[22] = xy[x]
                    _xy[23] = xy[x+1]
                #map jnt 13 to 12
                if x==13*2:
                    _xy[24] = xy[x]
                    _xy[25] = xy[x+1]         
                #map jnt 14 to 13
                if x==14*2:
                    _xy[26] = xy[x]
                    _xy[27] = xy[x+1]
                #map jnt 15 to 14
                if x==15*2:
                    _xy[28] = xy[x]
                    _xy[29] = xy[x+1]
                #map jnt 16 to 15
                if x==16*2:
                    _xy[30] = xy[x]
                    _xy[31] = xy[x+1]
                #map jnt 17 to 16
                if x==17*2:
                    _xy[32] = xy[x]
                    _xy[33] = xy[x+1]
                #map jnt 18 to 17
                if x==18*2:
                    _xy[34] = xy[x]
                    _xy[35] = xy[x+1]
            #coco 
            xy = _xy

        #add xy to frame
        cache[int(frame_indx[-1])] = xy

    # pdb.set_trace()DEBUG:
    plt.figure(1)
    #某个节点坐标的变化曲线
    drop_curves_plot = show_anim_curves(cache, plt)
    # pngName = os.path.join(os.path.dirname(os.path.dirname(__file__)),"test/gif_output/dirty_plot.png")
    # drop_curves_plot.savefig(pngName)
    # logger.info('writing gif_output/dirty_plot.png')

    # exit if no smoothing
    if not smooth:
        # return frames cache incl. 18 joints (x,y)
        return cache

    if len(json_files) == 1:
        logger.info("found single json file")
        # return frames cache incl. 18 joints (x,y) on single image\json
        return cache

    if len(json_files) <= 8:
        raise Exception("need more frames, min 9 frames/json files for smoothing!!!")

    logger.info("start smoothing")

    # create frame blocks
    head_frame_block = [int(re.findall("(\d+)", o)[-1]) for o in json_files[:4]]
    tail_frame_block = [int(re.findall("(\d+)", o)[-1]) for o in json_files[-4:]]

    ### smooth by median value, n frames 
    #通过临近 3 帧坐标中位数确定smooth坐标
    for frame, xy in cache.items():
        # create neighbor array based on frame index
        forward, back = ([] for _ in range(2))
        # pdb.set_trace()
        # joints x,y array
        _len = len(xy) # 36

        # create array of parallel frames (-3<n>3)
        for neighbor in range(1,4):
            # first n frames, get value of xy in postive lookahead frames(current frame + 3)
            if frame in head_frame_block:
                forward += cache[frame+neighbor]
            # last n frames, get value of xy in negative lookahead frames(current frame - 3)
            elif frame in tail_frame_block:
                back += cache[frame-neighbor]
            else:
                # between frames, get value of xy in bi-directional frames(current frame -+ 3)     
                forward += cache[frame+neighbor]
                back += cache[frame-neighbor]

        # build frame range vector 
        frames_joint_median = [0 for i in range(_len)]
        # more info about mapping in src/data_utils.py
        # for each 18joints*x,y  (x1,y1,x2,y2,...)~36 
        for x in range(0,_len,2):
            # set x and y
            y = x+1
            if frame in head_frame_block:
                # get vector of n frames forward for x and y, incl. current frame
                x_v = [xy[x], forward[x], forward[x+_len], forward[x+_len*2]]
                y_v = [xy[y], forward[y], forward[y+_len], forward[y+_len*2]]
            elif frame in tail_frame_block:
                # get vector of n frames back for x and y, incl. current frame
                x_v =[xy[x], back[x], back[x+_len], back[x+_len*2]]
                y_v =[xy[y], back[y], back[y+_len], back[y+_len*2]]
            else:
                # get vector of n frames forward/back for x and y, incl. current frame
                # median value calc: find neighbor frames joint value and sorted them, use numpy median module
                # frame[x1,y1,[x2,y2],..]frame[x1,y1,[x2,y2],...], frame[x1,y1,[x2,y2],..]
                #                 ^---------------------|-------------------------^
                x_v =[xy[x], forward[x], forward[x+_len], forward[x+_len*2],
                        back[x], back[x+_len], back[x+_len*2]]
                y_v =[xy[y], forward[y], forward[y+_len], forward[y+_len*2],
                        back[y], back[y+_len], back[y+_len*2]]

            # get median of vector
            x_med = np.median(sorted(x_v))
            y_med = np.median(sorted(y_v))

            # holding frame drops for joint
            if not x_med:
                # allow fix from first frame
                if frame:
                    # get x from last frame
                    x_med = smoothed[frame-1][x]
            # if joint is hidden y
            if not y_med:
                # allow fix from first frame
                if frame:
                    # get y from last frame
                    y_med = smoothed[frame-1][y]

            logger.debug("old X {0} sorted neighbor {1} new X {2}".format(xy[x],sorted(x_v), x_med))
            logger.debug("old Y {0} sorted neighbor {1} new Y {2}".format(xy[y],sorted(y_v), y_med))

            # build new array of joint x and y value
            frames_joint_median[x] = x_med 
            frames_joint_median[x+1] = y_med 
		

        smoothed[frame] = frames_joint_median

    return smoothed

def save3Djson(pose3d, frame):
    # pdb.set_trace()
    export_units = {}
    people = []
    first_people={}
    pose_keypoints_3d=[]
    P = np.array([0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27])
    vals = np.reshape( pose3d, (len(data_utils.H36M_NAMES), -1) )
    for i in P :
        for j in range(3):
            pose_keypoints_3d.append(vals[i,j])
    first_people["pose_keypoints_3d"]=pose_keypoints_3d
    people.append(first_people)
    export_units["version"]="3d-base-line"
    export_units["people"]=people
    
    # _out_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), outfilepath+'/{0}.json'.format(str(frame)))
    # with open(_out_file, 'w') as outfile:
    #     logger.info("exported maya json to {0}".format(_out_file))
    #     json.dump(export_units, outfile)
    
    return export_units




def main(_):
    #所有帧
    ALLjson={}
    #读json且smooth
    smoothed = read_openpose_json()
    #输出smooth后的坐标变化路径
    plt.figure(2)
    smooth_curves_plot = show_anim_curves(smoothed, plt)
    #return
    # pngName = os.path.join(os.path.dirname(os.path.dirname(__file__)),"test/gif_output/smooth_plot.png")
    # smooth_curves_plot.savefig(pngName)
    # logger.info('writing gif_output/smooth_plot.png')
    
    #插帧
    # pdb.set_trace()
    if FLAGS.interpolation:
        logger.info("start interpolation")

        framerange = len( smoothed.keys() )
        joint_rows = 36
        #将原来的dict转换为f*j的数组
        array = np.concatenate(list(smoothed.values()))
        array_reshaped = np.reshape(array, (framerange, joint_rows) )
        #插帧间隔，默认0.1
        multiplier = FLAGS.multiplier
        multiplier_inv = 1/multiplier

        out_array = np.array([])
        for row in range(joint_rows):
            #将全部的第 row 坐标插入x 
            x = []
            for frame in range(framerange):
                x.append( array_reshaped[frame, row] )
            # pdb.set_trace()
            frame = range( framerange )
            frame_resampled = np.arange(0, framerange, multiplier)
            #拟合曲线，k为平滑样条度数
            spl = UnivariateSpline(frame, x, k=3)
            #relative smooth factor based on jnt anim curve
            min_x, max_x = min(x), max(x)
            smooth_fac = max_x - min_x
            smooth_resamp = 125
            smooth_fac = smooth_fac * smooth_resamp
            spl.set_smoothing_factor( float(smooth_fac) )
            xnew = spl(frame_resampled)
            
            out_array = np.append(out_array, xnew)
    
        # pdb.set_trace()
        logger.info("done interpolating. reshaping {0} frames,  please wait!!".format(framerange))
    
        a = np.array([])
        for frame in range( int( framerange * multiplier_inv ) ):
            jnt_array = []
            for jnt in range(joint_rows):
                jnt_array.append( out_array[ jnt * int(framerange * multiplier_inv) + frame] )
            a = np.append(a, jnt_array)
        
        # pdb.set_trace()
        a = np.reshape(a, (int(framerange * multiplier_inv), joint_rows))
        out_array = a
    
        interpolate_smoothed = {}
        for frame in range( int(framerange * multiplier_inv) ):
            interpolate_smoothed[frame] = list( out_array[frame] )
        
        # pdb.set_trace()
        plt.figure(3)
        smoothed = interpolate_smoothed
        interpolate_curves_plot = show_anim_curves(smoothed, plt)
        # pngName = os.path.join(os.path.dirname(os.path.dirname(__file__)),"test/gif_output/interpolate_{0}.png".format(smooth_resamp))
        # interpolate_curves_plot.savefig(pngName)
        # logger.info('writing gif_output/interpolate_plot.png')

    #prediction
    # pdb.set_trace()
    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]

    actions = data_utils.define_actions(FLAGS.action)

    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    # pdb.set_trace()
    device_count = {"GPU": 0}
    png_lib = []
    before_pose = None
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
        #plt.figure(3)
        batch_size = 128
        model = create_model(sess, actions, batch_size)
        iter_range = len(smoothed.keys())
        export_units = {}
        twod_export_units = {}
        for n, (frame, xy) in enumerate(smoothed.items()):
            logger.info("calc frame {0}/{1}".format(frame, iter_range))
            # map list into np array  
            joints_array = np.zeros((1, 36))
            joints_array[0] = [0 for i in range(36)]
            for o in range(len(joints_array[0])):
                #feed array with xy array
                joints_array[0][o] = float(xy[o])

            twod_export_units[frame]={}
            for abs_b, __n in enumerate(range(0, len(xy),2)):
                twod_export_units[frame][abs_b] = {"translate": [xy[__n],xy[__n+1]]}

            _data = joints_array[0]
            # mapping all body parts or 3d-pose-baseline format
            for i in range(len(order)):
                for j in range(2):
                    # create encoder input
                    enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]
            for j in range(2):
                # Hip
                enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                # Neck/Nose
                enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
                # Thorax
                enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]

            # set spine
            spine_x = enc_in[0][24]
            spine_y = enc_in[0][25]

            enc_in = enc_in[:, dim_to_use_2d]
            mu = data_mean_2d[dim_to_use_2d]
            stddev = data_std_2d[dim_to_use_2d]
            enc_in = np.divide((enc_in - mu), stddev)

            dp = 1.0
            dec_out = np.zeros((1, 48))
            dec_out[0] = [0 for i in range(48)]
            _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
            all_poses_3d = []
            enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
            poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
            plt.axis('off')
            all_poses_3d.append( poses3d )
            enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
            subplot_idx, exidx = 1, 1
            _max = 0
            _min = 10000

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    tmp = poses3d[i][j * 3 + 2]
                    poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                    poses3d[i][j * 3 + 1] = tmp
                    if poses3d[i][j * 3 + 2] > _max:
                        _max = poses3d[i][j * 3 + 2]
                    if poses3d[i][j * 3 + 2] < _min:
                        _min = poses3d[i][j * 3 + 2]

            for i in range(poses3d.shape[0]):
                for j in range(32):
                    poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min
                    poses3d[i][j * 3] += (spine_x - 630)
                    poses3d[i][j * 3 + 2] += (500 - spine_y)

            # Plot 3d predictions
            ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
            ax.view_init(18, -70)    

            if FLAGS.cache_on_fail:
                if np.min(poses3d) < -1000:
                    poses3d = before_pose

            p3d = poses3d
            logger.info("frame score {0}".format(np.min(poses3d)))
            x,y,z = [[] for _ in range(3)]
            if not poses3d is None:
                to_export = poses3d.tolist()[0]
            else:
                to_export = [0.0 for _ in range(96)]
            logger.debug("export {0}".format(to_export))
            for o in range(0, len(to_export), 3):
                x.append(to_export[o])
                y.append(to_export[o+1])
                z.append(to_export[o+2])
            # pdb.set_trace()
            xx = p3d[0][0]
            yy = p3d[0][1]
            zz = p3d[0][2]
            for o in range(0, len(p3d[0]), 3):
                p3d[0][o] -= xx
                p3d[0][o+1] -= yy
                p3d[0][o+2] -= zz
            export_units[frame]={}
            # pdb.set_trace()
            for jnt_index, (_x, _y, _z) in enumerate(zip(x,y,z)):
                export_units[frame][jnt_index] = {"translate": [_x, _y, _z]}
               
            viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")
            ALLjson[str(frame)]=save3Djson(p3d , frame)
            # pngName = os.path.join(os.path.dirname(os.path.dirname(__file__)),'test\\gif_output\\pose_frame_{0}.png'.format(str(frame).zfill(12)))
            # # #TODO:
            # # # pdb.set_trace()
            # plt.savefig(pngName)
            # if FLAGS.write_gif:
            #     png_lib.append(imageio.imread(pngName))

            if FLAGS.cache_on_fail:
                before_pose = poses3d
    _out_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), outf)
    with open(_out_file, 'w') as outfile:
        logger.info("exported maya json to data.json")
        json.dump(ALLjson, outfile)

    # if FLAGS.write_gif:
    #     if FLAGS.interpolation:
    #         #take every frame on gif_fps * multiplier_inv
    #         png_lib = np.array([png_lib[png_image] for png_image in range(0,len(png_lib), int(multiplier_inv)) ])
    #     logger.info("creating Gif gif_output/animation.gif, please Wait!")
    #     gifName = os.path.join(os.path.dirname(os.path.dirname(__file__)),"test/gif_output/animation.gif")
    #     imageio.mimsave(gifName, png_lib, fps=FLAGS.gif_fps)
        # TODO:

    # _out_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'maya/3d_data.json')
    # twod_out_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'maya/2d_data.json')
    # with open(_out_file, 'w') as outfile:
    #     logger.info("exported maya json to {0}".format(_out_file))
    #     json.dump(export_units, outfile)
    # with open(twod_out_file, 'w') as outfile:
    #     logger.info("exported maya json to {0}".format(twod_out_file))
    #     json.dump(twod_export_units, outfile)
    

    logger.info("Done!")

if __name__ == "__main__":

    openpose_output_dir = FLAGS.pose_estimation_json
    
    level = {0:logging.ERROR,
             1:logging.WARNING,
             2:logging.INFO,
             3:logging.DEBUG}

    logger.setLevel(level[FLAGS.verbose])


    tf.app.run()
