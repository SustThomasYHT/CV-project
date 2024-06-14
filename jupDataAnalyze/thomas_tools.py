import os
import pandas as pd
import numpy as np
import cv2
import time
import tqdm
from joblib import dump, load
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

def merge_project(project_dir):
    if os.path.exists(os.path.join(project_dir, 'df-rawdata.csv')):
        return pd.read_csv(os.path.join(project_dir, 'df-rawdata.csv'))
    
    csv_dir = os.path.join(project_dir, 'rawcsv')
    csv_files = os.listdir(csv_dir)
    csv_files = [os.path.join(csv_dir, file) for file in csv_files if file.endswith('.csv') and '_' in file]
    csv_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # print(csv_files)
    
    df = pd.DataFrame()
    length = 0
    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file)
        if length == 0:
            length = len(temp_df)
        temp_df['Frame'] = temp_df['Frame'] + int(csv_file.split('_')[-1].split('.')[0])*length
        df = pd.concat([df, temp_df], ignore_index=True)
        
    df.to_csv(os.path.join(project_dir, 'df-rawdata.csv'), index=False)
    return df

def preprocess_project(project_dir):
    if os.path.exists(os.path.join(project_dir, 'df-preprocessed.csv')):
        return pd.read_csv(os.path.join(project_dir, 'df-preprocessed.csv'))
    
    df = pd.read_csv(os.path.join(project_dir, 'df-rawdata.csv'))
    
    #五个核心feature
    # 计算速度（距离变化量/时间变化量，这里时间变化量为1）
    df['Vx'] = df['X'].diff() # X方向速度
    df['Vy'] = df['Y'].diff() # Y方向速度
    # 计算加速度
    df['Ax'] = df['Vx'].diff() # X方向加速度
    df['Ay'] = df['Vy'].diff() # Y方向加速度
    # 计算方向变化，直接计算连续帧之间的角度变化
    df['Angle'] = np.degrees(np.arctan2(df['Vy'], df['Vx']))

    # df进行归一化
    for col in df.columns:
        if col not in ['Frame', 'Label', 'Visibility']:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            
    df.to_csv(os.path.join(project_dir, 'df-preprocessed.csv'), index=False)
    
    return df
    
def prepare_inference_df1(project_dir, shift_param=(15, 15)):
    df = pd.read_csv(os.path.join(project_dir, 'df-preprocessed.csv'))
    
    if os.path.exists(os.path.join(project_dir, f'df-inference-{shift_param}-old.csv')):
        return pd.read_csv(os.path.join(project_dir, f'df-inference-{shift_param}.csv')), df['Frame']

    # 去掉Frame和Label列
    features = df.drop(['Frame', 'Visibility', 'X', 'Y'], axis=1).values

    # 分别获取前、后shift参数
    front_shift_rows, back_shift_rows = shift_param
    total_shifts = front_shift_rows + back_shift_rows
    total_rows = len(features)

    # 创建一个新的数组来存放扩展后的特征
    expanded_features_list = []
    
    # 将数据进行填补，以处理边界的情况
    padded_features = np.pad(features, ((front_shift_rows, back_shift_rows), (0, 0)), mode='constant', constant_values=0)

    for index in range(total_rows):
        # 从填补后的数组中提取所需的窗口
        expanded_row = padded_features[index: index + total_shifts + 1].flatten()
        expanded_features_list.append(expanded_row)
        
    expanded_features = pd.DataFrame(expanded_features_list)
    
    expanded_features.to_csv(os.path.join(project_dir, f'df-inference-{shift_param}.csv'), index=False)

    return expanded_features, df['Frame']

def prepare_inference_df(project_dir, shift_param=(15, 15)):
    # 读取预处理的CSV文件
    df = pd.read_csv(os.path.join(project_dir, 'df-preprocessed.csv'))
    
    # 构建输出文件的路径
    output_file_path = os.path.join(project_dir, f'df-inference-{shift_param}.csv')
    
    # 如果文件已存在，直接读取并返回
    if os.path.exists(output_file_path):
        return pd.read_csv(output_file_path), df['Frame']
    
    # 获取特征数据
    features = df.drop(['Frame', 'Visibility', 'X', 'Y'], axis=1).values

    # 获取前、后shift参数
    front_shift_rows, back_shift_rows = shift_param
    total_shifts = front_shift_rows + back_shift_rows
    total_rows = len(features)

    # 创建一个新的数组（padded_features）以处理边界的情况，将前shift和后shift行数分别用0填充
    padded_features = np.pad(features, ((front_shift_rows, back_shift_rows), (0, 0)), mode='constant', constant_values=0)
    
    # 利用滑窗技术进行矢量化的特征扩展
    expanded_features = np.lib.stride_tricks.sliding_window_view(padded_features, (total_shifts + 1, features.shape[1]))
    
    # 重塑形状，从(n_rows, window_size, n_features)到(n_rows, window_size * n_features)
    expanded_features = expanded_features.squeeze(axis=1).reshape(total_rows, -1)
    
    # 转换为DataFrame
    expanded_features_df = pd.DataFrame(expanded_features)
    
    # 保存处理后的特征数据到CSV文件
    expanded_features_df.to_csv(output_file_path, index=False)
    
    # 返回处理后的特征数据和原始的Frame列
    return expanded_features_df, df['Frame']

def write_pickframe(project_name, predictions, shift_param=(15, 15), window=75, tag=''):
    project_dir = os.path.join('../projects', project_name)
    
    # 将预测结果predictions处理平滑
    i = 0
    while i < len(predictions):
        if predictions[i]==1 and sum(predictions[i:i+window])>1:
            predictions[i:i+window] = [1]*window
            i = i+window-1
        else:
            predictions[i] = 0
            i = i+1

    # 读取视频文件
    video_path = os.path.join("../video", project_name + ".mp4")
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 将预测结果写入视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out_path = os.path.join(project_dir, f'output-xgb-{shift_param}{tag}.avi')
    video_out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

    # 读取视频帧
    frame = 0
    tmp = time.time()
    for i in tqdm.tqdm(range(total_frames)):
        ret, img = cap.read()
        if not ret:
            break

        # 画上预测结果
        if frame < len(predictions):
            label = predictions[frame]
            frame += 1
            if label==1:
                video_out.write(img)
                continue
            else:
                continue
        else:
            break
        
    video_out.release()
    cap.release()
    cv2.destroyAllWindows()
    
def write_label(project_name, predictions, shift_param=(15, 15), tag=''):
    project_dir = os.path.join('../projects', project_name)

    # 读取视频文件
    video_path = os.path.join("../video", project_name + ".mp4")
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 将预测结果写入视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out_path = os.path.join(project_dir, f'output-xgb-{shift_param}-label{tag}.avi')
    video_out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

    # 读取视频帧
    frame = 0
    tmp = time.time()
    for i in tqdm.tqdm(range(total_frames)):
        ret, img = cap.read()
        if not ret:
            break

        # 画上预测结果
        if frame < len(predictions):
            label = predictions[frame]
            cv2.putText(img, str(label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            video_out.write(img)
        else:
            continue

        frame += 1
        
    video_out.release()
    cap.release()
    cv2.destroyAllWindows()

def get_label(project_name):
    df = pd.read_csv(os.path.join('../projects', project_name, 'df-traindata-all.csv'))
    return df['Label']

# def track_badminton(**args):
#     num_workers = args.get('batch_size', 2)
#     video_name = args.get('video_file').split('/')[-1][:-4]

#     output_vid = True

#     out_csv_file = os.path.join(args.get('save_dir'), f'{video_name}_ball.csv')
    

#     if not os.path.exists(args.get('save_dir')):
#         os.makedirs(args.get('save_dir'))

#     # Load model
#     tracknet_ckpt = torch.load(args.get('tracknet_file'))
#     tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
#     bg_mode = tracknet_ckpt['param_dict']['bg_mode']
#     tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
#     tracknet.load_state_dict(tracknet_ckpt['model'])

#     if args.get('inpaintnet_file'):
#         inpaintnet_ckpt = torch.load(args.get('inpaintnet_file'))
#         inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
#         inpaintnet = get_model('InpaintNet').cuda()
#         inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
#     else:
#         inpaintnet = None
        
#     cap = cv2.VideoCapture(args.get('video_file'))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f'Total frames: {total_frames}')
#     # input("Press Enter to continue...")
    
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     w, h = int(w * 480 / h), 480
#     first_frame = cap.read()[1]
#     first_frame = cv2.resize(first_frame, (w, h))
#     mask = create_mask_from_user_drawing(first_frame, draw=False)
    
#     range_length = args.get('range_length', 900)
#     start_time = time.time()
    
#     def torch_being_using():
#         with open('torch_being_using', 'r') as f:
#             return bool(int(f.read()))
        
#     def set_torch_being_using(val):
#         with open('torch_being_using', 'w') as f:
#             f.write(str(int(val)))
            
#     set_torch_being_using(False)
    
#     def process_frames(start, range_length, mask):
#         out_video_file = os.path.join(args.get('save_dir'), 'clips', f'{video_name}_{int(start//range_length)}.mp4')
#         if os.path.exists(out_video_file):
#             print(f'{out_video_file} already exists. Skipping...')
#             return
        
#         print(f'use time: {time.time() - start_time:.2f} sec')
#         # Sample all frames from video
#         frame_list, fps, (w, h) = generate_frames_range(args.get('video_file'), mask, start, range_length)
#         w_scaler, h_scaler = w / WIDTH, h / HEIGHT
#         img_scaler = (w_scaler, h_scaler)

#         print()
#         print(f'Processing {start} - {min(total_frames,start+range_length)} frames...')

#         # Test on TrackNet
#         tracknet.eval()
#         seq_len = tracknet_seq_len
#         tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
#                               'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}
#         if args.get('eval_mode') == 'nonoverlap':
#             tmp = time.time()
#             # Create dataset with non-overlap sampling
#             dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap',
#                                                      bg_mode=bg_mode,
#                                                      frame_arr=np.array(frame_list)[:, :, :, ::-1], padding=True)
#             print(f'dataset use time: {time.time() - tmp:.2f} sec')
#             data_loader = DataLoader(dataset, batch_size=args.get('batch_size'), shuffle=False, num_workers=num_workers,
#                                      drop_last=False)
#             print(f'data_loader use time: {time.time() - tmp:.2f} sec')
#             import random
#             time.sleep(random.random())
#             # while torch_being_using():
#             #     print(f'fuck {start} {torch_being_using()}')
#             #     time.sleep(random.random())
#             infer_badminton(img_scaler, tracknet_pred_dict, data_loader)

#         write_result(start, range_length, frame_list, fps, w, h, tracknet_pred_dict, None)
#         print('Done.')

#     def infer_badminton(img_scaler, tracknet_pred_dict, data_loader):
#         # set_torch_being_using(True)
#         for step, (i, x) in enumerate(tqdm(data_loader)):
#             x = x.float().cuda()
#             with torch.no_grad():
#                 y_pred = tracknet(x).detach().cpu()

#                 # Predict
#             tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
#             for key in tmp_pred.keys():
#                 tracknet_pred_dict[key].extend(tmp_pred[key])
#         # set_torch_being_using(False)

#     def write_result(start, range_length, frame_list, fps, w, h, tracknet_pred_dict, inpaint_pred_dict):
#         if not os.path.exists(os.path.join(args.get('save_dir'), 'rawcsv')):
#             os.makedirs(os.path.join(args.get('save_dir'), 'rawcsv'))
            
#         if not os.path.exists(os.path.join(args.get('save_dir'), 'clips')):
#             os.makedirs(os.path.join(args.get('save_dir'), 'clips'))
        
#         # Write csv file
#         pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
#         out_csv_file = os.path.join(args.get('save_dir'), 'rawcsv', f'{video_name}_{int(start//range_length)}.csv')
#         write_pred_csv(pred_dict, save_file=out_csv_file)

#         # Write video with predicted coordinates
#         if args.get('output_video'):
#             # w = frame_list[0].shape[1]
#             # h = frame_list[0].shape[0]
#             out_video_file = os.path.join(args.get('save_dir'), 'clips', f'{video_name}_{int(start//range_length)}.mp4')
#             print(f'Writing video: {out_video_file}')
#             write_pred_video(frame_list, dict(fps=fps, shape=(w, h)), pred_dict, save_file=out_video_file,
#                              traj_len=args.get('traj_len'))
    
#     max_workers = 1
#     print('wtf')
#     # Parallel(n_jobs=max_workers)(delayed(process_frames)(start, range_length, mask) for start in range(0, total_frames, range_length))
#     for start in range(0, total_frames, range_length):
#         process_frames(start, range_length, mask)






class Inferencer:
    def __init__(self, project_name, shift_param=(15, 15)):
        self.project_name = project_name
        self.project_dir = project_dir = os.path.join('../projects', project_name)
        self.shift_param = shift_param
        self.inference_df, self.frame = prepare_inference_df(self.project_dir, self.shift_param)
        
        
    def track_project(self):
        if not os.path.exists(self.project_dir):
            from TrackNetV3.thomas import track_badminton
            mode = 'nonoverlap'
            video_name = self.project_name
            video = f'../video/{video_name}.mp4'
            track_badminton(batch_size=4, eval_mode=mode, save_dir=f'../projects/{video_name}', tracknet_file='../ckpts/TrackNet_best.pt', video_file=video, output_video=True, traj_len=1, range_length=900) 
        else:
            print('Project badminton already been tracked.')
        
    def infer(self, model_type = 'all-marin'):
        self.track_project()

        merge_project(self.project_dir)
        preprocess_project(self.project_dir)
        self.inference_df, self.frame = prepare_inference_df(self.project_dir, self.shift_param)

        self.model_type = model_type
        model_path = os.path.join(f'./{model_type}', f'model-xgb-{self.shift_param}.joblib')
        self.model = load(model_path)
        self.predictions = self.model.predict(self.inference_df.values)
        return self.predictions
    
    def write_pickframe(self, window=75):
        if not hasattr(self, 'predictions'):
            self.predictions = self.infer()
        write_pickframe(self.project_name, self.predictions, self.shift_param, window=window, tag='-'+self.model_type)
        
    def write_label(self):
        if not hasattr(self, 'predictions'):
            self.predictions = self.infer()
        write_label(self.project_name, self.predictions, self.shift_param, tag='-'+self.model_type)