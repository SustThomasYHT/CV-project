import os
import argparse
import numpy as np
from tqdm import tqdm
import concurrent
import time
import torch
from torch.utils.data import DataLoader

from predict import predict
from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset
from utils.general import *

from joblib import Parallel, delayed, parallel_backend

def track_badminton(**args):
    track_start = time.time()

    num_workers = args.get('batch_size', 2)
    video_name = args.get('video_file').split('/')[-1][:-4]

    output_vid = True

    out_csv_file = os.path.join(args.get('save_dir'), f'{video_name}_ball.csv')
    

    if not os.path.exists(args.get('save_dir')):
        os.makedirs(args.get('save_dir'))

    # Load model
    tracknet_ckpt = torch.load(args.get('tracknet_file'))
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.get('inpaintnet_file'):
        inpaintnet_ckpt = torch.load(args.get('inpaintnet_file'))
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None
        
    cap = cv2.VideoCapture(args.get('video_file'))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames: {total_frames}')
    # input("Press Enter to continue...")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w, h = int(w * 480 / h), 480
    first_frame = cap.read()[1]
    first_frame = cv2.resize(first_frame, (w, h))
    mask = create_mask_from_user_drawing(first_frame, draw=False)
    
    range_length = args.get('range_length', 900)
    start_time = time.time()
    
    def torch_being_using():
        with open('torch_being_using', 'r') as f:
            return bool(int(f.read()))
        
    def set_torch_being_using(val):
        with open('torch_being_using', 'w') as f:
            f.write(str(int(val)))
            
    set_torch_being_using(False)
    
    def process_frames(start, range_length, mask):
        out_video_file = os.path.join(args.get('save_dir'), 'clips', f'{video_name}_{int(start//range_length)}.mp4')
        if os.path.exists(out_video_file):
            print(f'{out_video_file} already exists. Skipping...')
            return
        
        print(f'use time: {time.time() - start_time:.2f} sec')
        # Sample all frames from video
        frame_list, fps, (w, h) = generate_frames_range(args.get('video_file'), mask, start, range_length)
        w_scaler, h_scaler = w / WIDTH, h / HEIGHT
        img_scaler = (w_scaler, h_scaler)

        print()
        print(f'Processing {start} - {min(total_frames,start+range_length)} frames...')

        # Test on TrackNet
        tracknet.eval()
        seq_len = tracknet_seq_len
        tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                              'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}
        if args.get('eval_mode') == 'nonoverlap':
            tmp = time.time()
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap',
                                                     bg_mode=bg_mode,
                                                     frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
            print(f'dataset use time: {time.time() - tmp:.2f} sec')
            data_loader = DataLoader(dataset, batch_size=args.get('batch_size'), shuffle=False, num_workers=num_workers,
                                     drop_last=False)
            print(f'data_loader use time: {time.time() - tmp:.2f} sec')
            import random
            time.sleep(random.random())
            # while torch_being_using():
            #     print(f'fuck {start} {torch_being_using()}')
            #     time.sleep(random.random())
            infer_badminton(img_scaler, tracknet_pred_dict, data_loader)

        write_result(start, range_length, frame_list, fps, w, h, tracknet_pred_dict, None)
        print('Done.')

    def infer_badminton(img_scaler, tracknet_pred_dict, data_loader):
        # set_torch_being_using(True)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()

                # Predict
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
        # set_torch_being_using(False)

    def write_result(start, range_length, frame_list, fps, w, h, tracknet_pred_dict, inpaint_pred_dict):
        if not os.path.exists(os.path.join(args.get('save_dir'), 'rawcsv')):
            os.makedirs(os.path.join(args.get('save_dir'), 'rawcsv'))
            
        if not os.path.exists(os.path.join(args.get('save_dir'), 'clips')):
            os.makedirs(os.path.join(args.get('save_dir'), 'clips'))
        
        # Write csv file
        pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
        out_csv_file = os.path.join(args.get('save_dir'), 'rawcsv', f'{video_name}_{int(start//range_length)}.csv')
        write_pred_csv(pred_dict, save_file=out_csv_file)

        # Write video with predicted coordinates
        if args.get('output_video'):
            # w = frame_list[0].shape[1]
            # h = frame_list[0].shape[0]
            out_video_file = os.path.join(args.get('save_dir'), 'clips', f'{video_name}_{int(start//range_length)}.mp4')
            print(f'Writing video: {out_video_file}')
            write_pred_video(frame_list, dict(fps=fps, shape=(w, h)), pred_dict, save_file=out_video_file,
                             traj_len=args.get('traj_len'))
    
    # max_workers = 1
    # print('wtf')
    # Parallel(n_jobs=max_workers)(delayed(process_frames)(start, range_length, mask) for start in range(0, total_frames, range_length))
    for start in range(0, total_frames, range_length):
        process_frames(start, range_length, mask)
        
    with open(os.path.join(args.get('save_dir'), 'track_time.txt'), 'w') as f:
        f.write(f'{time.time() - track_start:.2f}')
    
if __name__ == '__main__':
    import time
    start = time.time()
    mode = 'nonoverlap'
    video_names = ['sunyu-lixuerui','sunyu-sindhu-cn','sunyu-yamaguchi']
    video_names = ['sunyu-intanon','sunyu-marin','sunyu-sindhu','sunyu-taitzuyin']
    video_names = ['test']
    
    video_names = os.listdir('video')
    video_names = [v[:-4] for v in video_names if v.startswith('20')]
    input(video_names)
    
    for video_name in video_names:
        video = f'video/{video_name}.mp4'
        # inpaintnet_file='ckpts/InpaintNet_best.pt',
        track_badminton(batch_size=4, eval_mode=mode, save_dir=f'projects/{video_name}', tracknet_file='ckpts/TrackNet_best.pt', video_file=video, output_video=True, traj_len=1, range_length=900)
        print(f'Elapsed time: {time.time() - start:.2f} sec')
    # input("Press Enter to continue...")