print("开始导入模块...")
import pandas as pd
print("导入pandas成功")
import numpy as np
print("导入numpy成功")
from pathlib import Path
print("导入pathlib成功")
from typing import Dict, List, Tuple
print("导入typing成功")
import joblib
print("导入joblib成功")
import xgboost as xgb
print("导入xgboost成功")
import torch
print("导入torch成功")
import lightgbm as lgb
print("导入lightgbm成功")
from sklearn.model_selection import KFold
import gc
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("开始导入本地模块...")
# 添加项目根目录到Python路径
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.data_loader import MarchMadnessDataLoader
print("导入data_loader成功")
from src.features.feature_engineering import FeatureEngineering
print("导入feature_engineering成功")
from src.models.model_trainer import ModelTrainer
print("导入model_trainer成功")

def prepare_training_data(
    data_loader: MarchMadnessDataLoader,
    feature_engineer: FeatureEngineering
) -> Tuple[np.ndarray, np.ndarray]:
    """准备训练数据"""
    print("\n=== 开始准备训练数据 ===")
    print("处理 2003 到 2024 赛季的数据")
    
    print("预加载数据...")
    print("1. 加载锦标赛结果...")
    tournament_results = data_loader.load_tournament_results()
    print(f"   - 加载了 {len(tournament_results)} 场锦标赛比赛")
    
    print("2. 加载种子排名数据...")
    tournament_seeds = data_loader.load_tournament_seeds()
    print(f"   - 加载了 {len(tournament_seeds)} 条种子排名数据")
    
    print("3. 加载常规赛结果...")
    regular_season_results = data_loader.load_regular_season_results()
    print(f"   - 加载了 {len(regular_season_results)} 场常规赛比赛")
    
    all_features = []
    all_labels = []
    total_games = 0
    
    for season in range(2003, 2025):
        print(f"\n--- 处理第 {season-2002}/{2024-2002} 个赛季 ({season}) ---")
        
        # 加载本赛季特征
        print("1. 加载赛季特征...")
        try:
            season_features = data_loader.load_season_features(season)
            print(f"   获取到 {len(season_features)} 支球队的数据")
            
            # 预先检查是否有缺失数据
            missing_cols = season_features.isnull().sum()
            if missing_cols.any():
                print("   警告: 发现缺失数据:")
                for col, count in missing_cols[missing_cols > 0].items():
                    print(f"   - {col}: {count} 个缺失值")
        except Exception as e:
            print(f"   错误: 加载赛季特征失败 - {str(e)}")
            continue
        
        # 获取本赛季锦标赛比赛
        print("2. 筛选本赛季比赛...")
        season_games = tournament_results[
            tournament_results["Season"] == season
        ]
        n_games = len(season_games)
        print(f"   找到 {n_games} 场比赛")
        total_games += n_games
        
        print("3. 生成对抗特征...")
        games_processed = 0
        start_time = pd.Timestamp.now()
        
        for _, game in season_games.iterrows():
            team1_id = game["WTeamID"]  # 胜者ID
            team2_id = game["LTeamID"]  # 败者ID
            
            # 获取两队的赛季统计数据
            team1_stats = season_features[
                season_features["TeamID"] == team1_id
            ]
            team2_stats = season_features[
                season_features["TeamID"] == team2_id
            ]
            
            if len(team1_stats) == 0 or len(team2_stats) == 0:
                print(f"   警告: 找不到球队 {team1_id} 或 {team2_id} 的数据")
                continue
            
            try:
                # 创建对抗特征
                matchup_features = feature_engineer.create_matchup_features(
                    team1_stats,
                    team2_stats
                )
                
                # 添加历史交手记录
                matchup_features = feature_engineer.add_historical_matchup_features(
                    matchup_features,
                    regular_season_results,
                    team1_id,
                    team2_id
                )
                
                # 添加种子排名特征
                matchup_features = feature_engineer.add_tournament_seed_features(
                    matchup_features,
                    tournament_seeds,
                    team1_id,
                    team2_id,
                    season
                )
                
                # 添加动量特征
                matchup_features = feature_engineer.add_momentum_features(
                    matchup_features,
                    regular_season_results,
                    team1_id,
                    team2_id
                )
                
                # 确保特征是一维数组
                feature_vector = matchup_features.values.flatten()
                if len(all_features) > 0:
                    if len(feature_vector) != len(all_features[0]):
                        print(f"   警告: 特征维度不匹配! 当前: {len(feature_vector)}, 期望: {len(all_features[0])}")
                        print(f"   特征列名: {matchup_features.columns.tolist()}")
                        continue
                all_features.append(feature_vector)
                all_labels.append(1)  # 胜者标签为1
                
                # 为同一场比赛创建反向特征(交换两队)
                matchup_features = feature_engineer.create_matchup_features(
                    team2_stats,
                    team1_stats
                )
                matchup_features = feature_engineer.add_historical_matchup_features(
                    matchup_features,
                    regular_season_results,
                    team2_id,
                    team1_id
                )
                matchup_features = feature_engineer.add_tournament_seed_features(
                    matchup_features,
                    tournament_seeds,
                    team2_id,
                    team1_id,
                    season
                )
                matchup_features = feature_engineer.add_momentum_features(
                    matchup_features,
                    regular_season_results,
                    team2_id,
                    team1_id
                )
                
                feature_vector = matchup_features.values.flatten()
                if len(all_features) > 0:
                    if len(feature_vector) != len(all_features[0]):
                        print(f"   警告: 特征维度不匹配! 当前: {len(feature_vector)}, 期望: {len(all_features[0])}")
                        continue
                all_features.append(feature_vector)
                all_labels.append(0)  # 败者标签为0
                
            except Exception as e:
                print(f"   错误: 处理比赛 {team1_id} vs {team2_id} 时失败 - {str(e)}")
                continue
            
            games_processed += 1
            if games_processed % 10 == 0:
                elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                speed = games_processed / elapsed if elapsed > 0 else 0
                remaining = (n_games - games_processed) / speed if speed > 0 else 0
                print(f"   已处理 {games_processed}/{n_games} 场比赛 "
                      f"(速度: {speed:.1f} 场/秒, 预计还需 {remaining:.1f} 秒)")
    
    print(f"\n=== 数据准备完成 ===")
    print(f"总共处理了 {total_games} 场比赛")
    print(f"收集了 {len(all_features)} 个样本")
    
    if len(all_features) == 0:
        raise ValueError("没有生成任何有效的特征!")
        
    return np.array(all_features), np.array(all_labels)

def make_predictions(
    models: Dict,
    features: np.ndarray,
    model_name: str = None
) -> np.ndarray:
    """集成模型或单个模型预测"""
    # 应用特征选择
    if "feature_mask" in models:
        features = features[:, models["feature_mask"]]
    
    if model_name is not None:
        # 单个模型预测
        if model_name == "lightgbm":
            return models["lightgbm"].predict(features)
        elif model_name == "xgboost":
            return models["xgboost"].predict(xgb.DMatrix(features))
        elif model_name == "neural_network":
            models["neural_network"].eval()
            with torch.no_grad():
                return models["neural_network"](
                    torch.FloatTensor(features)
                ).cpu().numpy()
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
    
    # 获取各个模型的预测
    lgb_pred = models["lightgbm"].predict(features)
    xgb_pred = models["xgboost"].predict(xgb.DMatrix(features))
    
    models["neural_network"].eval()
    with torch.no_grad():
        nn_pred = models["neural_network"](
            torch.FloatTensor(features)
        ).cpu().numpy()
    
    # 使用动态权重集成
    if "ensemble_weights" in models:
        weights = models["ensemble_weights"]
        ensemble_pred = (
            weights[0] * lgb_pred +
            weights[1] * xgb_pred +
            weights[2] * nn_pred
        )
    else:
        # 使用Stacking模型
        base_predictions = np.column_stack([
            lgb_pred,
            xgb_pred,
            nn_pred
        ])
        ensemble_pred = models["stacking"].predict(base_predictions)
    
    return ensemble_pred

def calculate_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Brier Score"""
    return np.mean((y_pred - y_true) ** 2)

def train_models(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray
) -> Dict:
    """训练多个模型并评估性能"""
    print("\n=== 开始训练模型 ===")
    trainer = ModelTrainer(
        train_features,
        train_labels,
        val_features,
        val_labels
    )
    
    models = {}
    brier_scores = {}
    
    # 特征重要性分析
    print("\n--- 特征重要性分析 ---")
    feature_importance_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        n_jobs=1
    )
    feature_importance_model.fit(train_features, train_labels)
    importances = feature_importance_model.feature_importances_
    
    # 选择重要特征
    importance_threshold = np.percentile(importances, 20)  # 保留前80%的特征
    important_features = importances >= importance_threshold
    print(f"选择了 {sum(important_features)}/{len(importances)} 个重要特征")
    
    # 使用重要特征训练模型
    train_features_selected = train_features[:, important_features]
    val_features_selected = val_features[:, important_features]
    
    print("\n--- 训练LightGBM模型 ---")
    lgb_params, lgb_model = trainer.optimize_lightgbm_params(n_trials=10)
    models["lightgbm"] = lgb_model
    models["lgb_params"] = lgb_params
    
    # 评估LightGBM
    lgb_pred = make_predictions(models, val_features_selected, "lightgbm")
    brier_scores["lightgbm"] = calculate_brier_score(val_labels, lgb_pred)
    print(f"LightGBM Brier Score: {brier_scores['lightgbm']:.4f}")
    
    print("\n--- 训练XGBoost模型 ---")
    xgb_model = trainer.train_xgboost()
    models["xgboost"] = xgb_model
    
    # 评估XGBoost
    xgb_pred = make_predictions(models, val_features_selected, "xgboost")
    brier_scores["xgboost"] = calculate_brier_score(val_labels, xgb_pred)
    print(f"XGBoost Brier Score: {brier_scores['xgboost']:.4f}")
    
    print("\n--- 训练神经网络模型 ---")
    nn_model = trainer.train_neural_network(
        input_dim=train_features_selected.shape[1],
        hidden_dims=[128, 64, 32],
        batch_size=128,
        num_epochs=100,
        learning_rate=0.001,
        weight_decay=0.01,
        dropout_rate=0.2
    )
    models["neural_network"] = nn_model
    
    # 评估神经网络
    nn_pred = make_predictions(models, val_features_selected, "neural_network")
    brier_scores["neural_network"] = calculate_brier_score(val_labels, nn_pred)
    print(f"神经网络 Brier Score: {brier_scores['neural_network']:.4f}")
    
    # 训练Stacking模型
    print("\n--- 训练Stacking模型 ---")
    base_predictions = np.column_stack([
        lgb_pred,
        xgb_pred,
        nn_pred
    ])
    
    stacking_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.01,
        num_leaves=31,
        n_jobs=1
    )
    
    # 使用5折交叉验证训练Stacking模型
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stacking_preds = np.zeros_like(val_labels)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(base_predictions)):
        print(f"训练第 {fold+1} 折...")
        X_train = base_predictions[train_idx]
        y_train = val_labels[train_idx]
        X_val = base_predictions[val_idx]
        
        stacking_model.fit(X_train, y_train)
        stacking_preds[val_idx] = stacking_model.predict(X_val)
    
    models["stacking"] = stacking_model
    brier_scores["stacking"] = calculate_brier_score(
        val_labels,
        stacking_preds
    )
    print(f"Stacking Brier Score: {brier_scores['stacking']:.4f}")
    
    # 动态权重集成
    print("\n--- 计算动态集成权重 ---")
    weights = np.array([
        1 / brier_scores["lightgbm"],
        1 / brier_scores["xgboost"],
        1 / brier_scores["neural_network"]
    ])
    weights = weights / np.sum(weights)
    print(f"集成权重: LightGBM={weights[0]:.3f}, XGBoost={weights[1]:.3f}, NN={weights[2]:.3f}")
    
    ensemble_pred = (
        weights[0] * lgb_pred +
        weights[1] * xgb_pred +
        weights[2] * nn_pred
    )
    
    brier_scores["ensemble"] = calculate_brier_score(
        val_labels,
        ensemble_pred
    )
    print(f"动态集成 Brier Score: {brier_scores['ensemble']:.4f}")
    
    print("\n=== Brier Score 评估结果 ===")
    for model_name, score in brier_scores.items():
        print(f"{model_name}: {score:.4f}")
    
    print("\n=== 模型训练完成 ===")
    models["brier_scores"] = brier_scores
    models["feature_mask"] = important_features
    models["ensemble_weights"] = weights
    return models

def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 初始化特征工程
    feature_eng = FeatureEngineering()
    
    print("\n=== 数据准备开始 ===")
    
    try:
        # 加载和处理数据
        features, labels = feature_eng.process_data()
        
        # 数据集分割
        print("\n=== 数据集分割 ===")
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42,
            shuffle=True, stratify=labels  # 确保标签分布一致
        )
        
        # 释放原始数据的内存
        del features, labels
        gc.collect()
        
        print(f"\n=== 数据集信息 ===")
        print(f"特征维度: {X_train.shape[1]}")
        print(f"训练集样本数: {len(X_train)}")
        print(f"验证集样本数: {len(X_val)}")
        print(f"正样本比例: {np.mean(y_train):.2%}")
        
        # 特征标准化
        print("\n=== 特征标准化 ===")
        scaler = StandardScaler()
        print("1. 标准化训练集...")
        X_train = scaler.fit_transform(X_train)
        print("2. 标准化验证集...")
        X_val = scaler.transform(X_val)
        
        print("\n=== 开始训练模型 ===")
        
        # 特征重要性分析
        print("\n--- 特征重要性分析 ---")
        lgb_model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1  # 单线程以减少内存使用
        )
        
        print("训练特征选择模型...")
        lgb_model.fit(X_train, y_train)
        
        # 选择重要特征
        importances = lgb_model.feature_importances_
        importance_threshold = np.percentile(importances, 20)
        selected_features = importances >= importance_threshold
        print(f"选择了 {sum(selected_features)}/{len(selected_features)} 个重要特征")
        
        # 使用选定的特征
        X_train = X_train[:, selected_features]
        X_val = X_val[:, selected_features]
        
        # 清理特征选择模型
        del lgb_model
        gc.collect()
        
        print("\n--- 训练最终模型 ---")
        
        # 优化后的模型参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 5,
            'min_child_samples': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'n_jobs': 1  # 单线程以减少内存使用
        }
        
        # 创建和训练模型
        print("训练LightGBM模型...")
        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=10)
            ]
        )
        
        # 评估模型
        print("\n=== 模型评估 ===")
        val_pred = model.predict_proba(X_val)[:, 1]
        brier_score = np.mean((val_pred - y_val) ** 2)
        print(f"验证集 Brier Score: {brier_score:.4f}")
        
        # 保存模型
        print("\n=== 保存模型 ===")
        if not Path('models').exists():
            Path('models').mkdir(parents=True)
            
        joblib.dump(model, 'models/lightgbm_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(selected_features, 'models/selected_features.pkl')
        
        print("\n=== 训练完成 ===")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        raise
    finally:
        # 清理内存
        gc.collect()

if __name__ == "__main__":
    main() 