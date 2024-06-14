import json
import os
import pandas as pd
import numpy as np
from thomas_tools import Inferencer, get_label



def try_shift(shift_param, device="cuda:0",project_names = ["sunyu-intanon", "sunyu-sindhu", "sunyu-marin"]):
    print(f'Trying shift param: {shift_param}')
    
    if os.path.exists(f"./new-model/model-xgb-{shift_param}.joblib"):
        print(f"Model for shift param {shift_param} exists. Skip.")
        return
    
    from sklearn.model_selection import train_test_split

    features = None
    labels = None
    for project_name in project_names:
        inferencer = Inferencer(project_name, shift_param)
        tmp_features = inferencer.inference_df
        tmp_labels = get_label(project_name)
        
        print(f'Project {project_name} features, labels shape: {tmp_features.shape} {tmp_labels.shape}')
        
        # features在每一行后追加
        if features is None:
            features = tmp_features
            labels = tmp_labels
        else:
            features = np.concatenate([features, tmp_features], axis=0)
            labels = np.concatenate([labels, tmp_labels], axis=0)
            
    print(f'features, labels shape: {features.shape} {labels.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.1, random_state=45
    )
    print(f'X_train shape: {X_train.shape}')

    from xgboost import XGBClassifier

    model = XGBClassifier(
        device=device,
        n_estimators=1000,
        random_state=42,
        use_label_encoder=False,
        eval_metric="aucpr",
    )

    model.fit(X_train, y_train)
    print(f'Model fitted.')

    predictions = model.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # 评估模型
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    precision = precision_score(y_test, predictions)
    print(f"Precision: {precision}")
    recall = recall_score(y_test, predictions)
    print(f"Recall: {recall}")
    f1 = f1_score(y_test, predictions)
    print(f"F1 score: {f1}")

    from joblib import dump, load

    model_path = os.path.join("./new-model", f"model-xgb-{shift_param}.joblib")
    dump(model, model_path)
    print(f"Model saved to {model_path}")

    return accuracy, precision, recall, f1


def try_shifts(shift_params):
    result_path = "./new-model/shift_results.json"
    for shift_param in shift_params:
        result = try_shift(shift_param)
        
        if result:
            # 打开result_path文件，更新结果
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    results = json.load(f)
            else:
                results = {}
                
            results[str(shift_param)] = result
            
            with open(result_path, "w") as f:
                json.dump(results, f, indent=4)
            

if __name__ == "__main__":
    shift_params = [
        (50, 50),
        (100, 100),
        (150, 150),
        (200, 200),
        (250, 250),
        (300, 300),
        (350, 350),
        (400, 400)
        ]
    # 每隔25尝试一次,例如（25，25），（50，50），（75，75）...
    shift_params = [(i, i) for i in range(25, 401, 25)]

    results = try_shifts(shift_params)
