import os
import pandas as pd
import numpy as np
import cv2
import time
import tqdm
from joblib import dump, load

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
    project_dir = os.path.join('./projects', project_name)
    
    # 将预测结果predictions处理平滑
    i = 0
    while i < len(predictions):
        if predictions[i]==1 and sum(predictions[i:i+window])>1:
            predictions[i:i+2] = [1]*2
            i = i+1
        else:
            predictions[i] = 0
            i = i+1

    # 读取视频文件
    video_path = os.path.join("./video", project_name + ".mp4")
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 将预测结果写入视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out_path = os.path.join(project_dir, f'output-xgb{tag}-{shift_param}-{window}.avi')
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
    project_dir = os.path.join('./projects', project_name)

    # 读取视频文件
    video_path = os.path.join("./video", project_name + ".mp4")
    cap = cv2.VideoCapture(video_path)

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 将预测结果写入视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out_path = os.path.join(project_dir, f'output-xgb{tag}-{shift_param}-label.avi')
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
    df = pd.read_csv(os.path.join('./projects', project_name, 'df-traindata-all.csv'))
    return df['Label']

class Inferencer:
    def __init__(self, project_name, shift_param=(15, 15)):
        self.project_name = project_name
        self.project_dir = project_dir = os.path.join('./projects', project_name)
        self.shift_param = shift_param
        print(f'Project name: {project_name} Shift param: {shift_param}')
        
    def track_project(self):
        if not os.path.exists(self.project_dir) or not os.path.exists(os.path.join(self.project_dir, 'clips')):
            from thomas import track_badminton
            mode = 'nonoverlap'
            video_name = self.project_name
            video = f'./video/{video_name}.mp4'
            track_badminton(batch_size=4, eval_mode=mode, save_dir=f'./projects/{video_name}', tracknet_file='./ckpts/TrackNet_best.pt', video_file=video, output_video=True, traj_len=1, range_length=900) 
        else:
            print('Project badminton already been tracked.')
        
    def infer(self, model_type = 'all-marin'):
        self.track_project()
        self.model_type = model_type
        predictions_csv = os.path.join(self.project_dir, f'df-predicted-{self.model_type}-{self.shift_param}.csv')
        if os.path.exists(predictions_csv):
            print(f'Predictions already exist: {predictions_csv}')
            self.predictions = pd.read_csv(predictions_csv).drop('Frame', axis=1).values
            self.predictions = self.predictions.flatten()
            return self.predictions

        print(f'Inferencing with model: {model_type}')
        merge_project(self.project_dir)
        preprocess_project(self.project_dir)
        self.inference_df, self.frame = prepare_inference_df(self.project_dir, self.shift_param)

        
        model_path = os.path.join(f'./jupDataAnalyze',model_type, f'model-xgb-{self.shift_param}.joblib')
        self.model = load(model_path)
        self.predictions = self.model.predict(self.inference_df.values)
        
        df_predictions = pd.DataFrame({'Frame': self.frame, 'Label': self.predictions})
        df_predictions.to_csv(predictions_csv, index=False)
        
        return self.predictions
    
    def write_pickframe(self, window=75):
        if not hasattr(self, 'predictions'):
            self.predictions = self.infer()
        
        print(self.predictions)
        write_pickframe(self.project_name, self.predictions, self.shift_param, window=window, tag='-'+self.model_type)
        
    def write_testpickframe(self, window=75):
        test_predictions = pd.read_csv(os.path.join(self.project_dir, 'df-rawdata.csv'))['Visibility'].values
        write_pickframe(self.project_name, test_predictions, self.shift_param, window=window, tag='-test')
        
    def write_label(self):
        if not hasattr(self, 'predictions'):
            self.predictions = self.infer()
        write_label(self.project_name, self.predictions, self.shift_param, tag='-'+self.model_type)