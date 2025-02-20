import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import TimeSeriesSplit, KFold
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import optuna
import gc

class TournamentDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None
    ):
        print(f"\n调试 TournamentDataset 初始化:")
        print(f"输入特征类型: {type(features)}, 形状: {features.shape}")
        if labels is not None:
            print(f"输入标签类型: {type(labels)}, 形状: {labels.shape}")
        
        # 检查特征值是否有效
        print("检查特征值...")
        if np.isnan(features).any():
            print("警告: 特征中包含 NaN 值，将替换为 0")
            features = np.nan_to_num(features, 0)
        if np.isinf(features).any():
            print("警告: 特征中包含 Inf 值，将替换为很大的数")
            features = np.clip(features, -1e9, 1e9)
            
        # 检查标签值是否有效
        if labels is not None:
            print("检查标签值...")
            if np.isnan(labels).any():
                raise ValueError("标签中包含 NaN 值")
            if np.isinf(labels).any():
                raise ValueError("标签中包含 Inf 值")
            if not np.all(np.logical_or(labels == 0, labels == 1)):
                raise ValueError("标签值不是二元的 (0 或 1)")
        
        try:
            print("转换特征为 FloatTensor...")
            # 确保数据类型是float32以节省内存
            if features.dtype != np.float32:
                print(f"将特征从 {features.dtype} 转换为 float32...")
                features = features.astype(np.float32)
            
            # 分批转换以避免内存问题
            batch_size = 1000
            n_samples = len(features)
            self.features = []
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                print(f"转换特征批次 {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}...")
                batch = torch.from_numpy(features[i:end_idx])
                self.features.append(batch)
            
            self.features = torch.cat(self.features)
            print(f"特征转换成功，形状: {self.features.shape}")
            
            if labels is not None:
                print("转换标签为 FloatTensor...")
                if labels.dtype != np.float32:
                    labels = labels.astype(np.float32)
                self.labels = torch.from_numpy(labels)
                print(f"标签转换成功，形状: {self.labels.shape}")
            else:
                self.labels = None
                
        except Exception as e:
            print(f"数据转换失败: {str(e)}")
            import traceback
            print("详细错误信息:")
            print(traceback.format_exc())
            raise
        
    def __len__(self) -> int:
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor
    ]:
        try:
            if self.labels is not None:
                return self.features[idx], self.labels[idx]
            return self.features[idx]
        except Exception as e:
            print(f"获取数据失败，索引 {idx}: {str(e)}")
            raise

class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim)
        weights = self.attention(x)  # (batch_size, seq_len, 1)
        return torch.sum(weights * x, dim=1)  # (batch_size, dim)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.prelu = nn.PReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(x + self.block(x))

class TournamentNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        num_residual_blocks: int = 2,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # 特征提取层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性变换
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity='leaky_relu')
            nn.init.constant_(linear.bias, 0)
            
            # 添加基本层
            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout_rate)
            ])
            
            # 添加残差块
            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(hidden_dim, dropout_rate))
            
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化输出层
        nn.init.xavier_normal_(self.output_layer[0].weight)
        nn.init.constant_(self.output_layer[0].bias, 0)
        
        self.debug = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug:
            print("\n=== 前向传播调试信息 ===")
            print(f"输入张量形状: {x.shape}")
            print(f"输入张量范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"输入张量均值: {x.mean().item():.4f}")
            print(f"输入张量标准差: {x.std().item():.4f}")
            
            # 输入标准化
            x = self.input_norm(x)
            x = self.input_dropout(x)
            print("\n输入标准化后:")
            print(f"范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"均值: {x.mean().item():.4f}")
            print(f"标准差: {x.std().item():.4f}")
            
            # 特征提取
            x = self.feature_extractor(x)
            print("\n特征提取后:")
            print(f"形状: {x.shape}")
            print(f"范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"均值: {x.mean().item():.4f}")
            print(f"标准差: {x.std().item():.4f}")
            
            # 输出层
            x = self.output_layer(x)
            print("\n输出层:")
            print(f"形状: {x.shape}")
            print(f"范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"均值: {x.mean().item():.4f}")
            print(f"标准差: {x.std().item():.4f}")
            
            return x.squeeze()
        else:
            x = self.input_norm(x)
            x = self.input_dropout(x)
            x = self.feature_extractor(x)
            x = self.output_layer(x)
            return x.squeeze()

class ModelTrainer:
    def __init__(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ):
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
    def train_lightgbm(self, params: Dict) -> lgb.Booster:
        """
        训练 LightGBM 模型
        """
        print("\n=== 训练 LightGBM 模型 ===")
        print("1. 准备训练数据...")
        
        # 转换数据类型以节省内存
        train_features = self.train_features.astype(np.float32)
        train_labels = self.train_labels.astype(np.float32)
        
        if self.val_features is not None:
            val_features = self.val_features.astype(np.float32)
            val_labels = self.val_labels.astype(np.float32)
        
        print("2. 创建数据集...")
        train_data = lgb.Dataset(
            train_features,
            label=train_labels,
            free_raw_data=True  # 释放原始数据
        )
        
        if self.val_features is not None:
            val_data = lgb.Dataset(
                val_features,
                label=val_labels,
                free_raw_data=True,
                reference=train_data
            )
        else:
            val_data = None
        
        print("3. 设置回调函数...")
        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(10)
        ]
        
        print("4. 开始训练...")
        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,  # 减少迭代次数
                valid_sets=[val_data] if val_data else None,
                callbacks=callbacks
            )
            print("5. 训练完成!")
            return model
        except Exception as e:
            print(f"训练过程中出错: {str(e)}")
            raise
    
    def train_xgboost(
        self,
        params: Optional[Dict] = None
    ) -> xgb.Booster:
        """训练XGBoost模型"""
        print("初始化XGBoost模型参数...")
        if params is None:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "min_child_weight": 1,
                "tree_method": "hist"
            }
            
        print("创建训练数据...")
        train_data = xgb.DMatrix(
            self.train_features,
            label=self.train_labels
        )
        
        if self.val_features is not None and self.val_labels is not None:
            print("创建验证集数据...")
            val_data = xgb.DMatrix(
                self.val_features,
                label=self.val_labels
            )
            print("开始训练XGBoost模型（带验证集）...")
            model = xgb.train(
                params,
                train_data,
                num_boost_round=1000,
                evals=[(train_data, "train"), (val_data, "val")],
                early_stopping_rounds=50,
                verbose_eval=10  # 每10轮输出一次日志
            )
            print(f"训练完成，最佳迭代轮数: {model.best_iteration}")
        else:
            print("开始训练XGBoost模型（无验证集）...")
            model = xgb.train(
                params,
                train_data,
                num_boost_round=1000,
                verbose_eval=10  # 每10轮输出一次日志
            )
            print("训练完成！")
            
        return model
    
    def train_neural_network(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        batch_size: int = 128,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        dropout_rate: float = 0.1,
        num_residual_blocks: int = 2
    ) -> TournamentNN:
        """训练神经网络模型"""
        print("\n=== 神经网络训练初始化 ===")
        print("检查输入数据:")
        print(f"训练特征形状: {self.train_features.shape}")
        print(f"训练标签形状: {self.train_labels.shape}")
        if self.val_features is not None:
            print(f"验证特征形状: {self.val_features.shape}")
            print(f"验证标签形状: {self.val_labels.shape}")
        
        print("\n检查数据类型和值范围:")
        print(f"训练特征类型: {self.train_features.dtype}")
        print(f"训练特征范围: [{np.min(self.train_features):.4f}, {np.max(self.train_features):.4f}]")
        print(f"训练标签类型: {self.train_labels.dtype}")
        print(f"训练标签范围: [{np.min(self.train_labels):.4f}, {np.max(self.train_labels):.4f}]")
        
        print("\n1. 创建数据加载器...")
        try:
            train_dataset = TournamentDataset(
                self.train_features,
                self.train_labels
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            
            if self.val_features is not None and self.val_labels is not None:
                val_dataset = TournamentDataset(
                    self.val_features,
                    self.val_labels
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    num_workers=0
                )
            else:
                val_loader = None
            
        except Exception as e:
            print(f"错误：创建数据加载器失败 - {str(e)}")
            raise
        
        print("\n2. 构建神经网络模型...")
        try:
            model = TournamentNN(
                input_dim,
                hidden_dims,
                num_residual_blocks,
                dropout_rate
            ).to(self.device)
        except Exception as e:
            print(f"错误：创建模型失败 - {str(e)}")
            raise
        
        print("\n3. 初始化优化器和损失函数...")
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            criterion = nn.BCELoss()
        except Exception as e:
            print(f"错误：创建优化器或损失函数失败 - {str(e)}")
            raise
        
        best_val_loss = float("inf")
        patience = 15
        counter = 0
        best_model = None
        
        print("\n=== 开始训练 ===")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练阶段
            model.train()
            train_loss = 0
            batch_count = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                if batch_count % 10 == 0:
                    print(f"Batch {batch_count}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
            train_loss /= len(train_loader)
            
            # 验证阶段
            if val_loader is not None:
                model.eval()
                val_loss = 0
                val_preds = []
                val_true = []
                
                with torch.no_grad():
                    for batch_features, batch_labels in val_loader:
                        batch_features = batch_features.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_labels)
                        val_loss += loss.item()
                        
                        val_preds.extend(outputs.cpu().numpy())
                        val_true.extend(batch_labels.cpu().numpy())
                        
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # 计算验证集的Brier Score
                val_preds = np.array(val_preds)
                val_true = np.array(val_true)
                brier_score = np.mean((val_preds - val_true) ** 2)
                
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Brier Score: {brier_score:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    best_model = model.state_dict()
                    print(f"验证损失改善！新的最佳损失: {best_val_loss:.4f}")
                else:
                    counter += 1
                    print(f"验证损失未改善，已经 {counter}/{patience} 个 epoch 无改善")
                    
                if counter >= patience:
                    print(f"\n早停：{patience} 个 epoch 验证损失未改善")
                    break
            else:
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
        if best_model is not None:
            model.load_state_dict(best_model)
        
        return model
    
    def optimize_lightgbm_params(self, n_trials=10):
        """使用 Optuna 优化 LightGBM 参数"""
        print("\n=== 开始 LightGBM 参数优化 ===")
        
        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 2, 20),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "max_bin": trial.suggest_int("max_bin", 128, 256),
                "n_jobs": 1
            }
            
            # 对于小样本量，使用留一法交叉验证
            if len(self.train_features) < 5:
                cv_scores = []
                for i in range(len(self.train_features)):
                    train_idx = [j for j in range(len(self.train_features)) if j != i]
                    val_idx = [i]
                    
                    X_train = self.train_features[train_idx].astype(np.float32)
                    y_train = self.train_labels[train_idx].astype(np.float32)
                    X_val = self.train_features[val_idx].astype(np.float32)
                    y_val = self.train_labels[val_idx].astype(np.float32)
                    
                    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
                    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=True)
                    
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=100,
                        valid_sets=[val_data],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=10),
                            lgb.log_evaluation(0)
                        ]
                    )
                    
                    val_preds = model.predict(X_val)
                    brier_score = np.mean((val_preds - y_val) ** 2)
                    cv_scores.append(brier_score)
                    
                    # 清理内存
                    del train_data, val_data, model
                    gc.collect()
            else:
                # 使用3折交叉验证
                cv_scores = []
                kf = KFold(n_splits=3, shuffle=True, random_state=42)
                
                for train_idx, val_idx in kf.split(self.train_features):
                    X_train = self.train_features[train_idx].astype(np.float32)
                    y_train = self.train_labels[train_idx].astype(np.float32)
                    X_val = self.train_features[val_idx].astype(np.float32)
                    y_val = self.train_labels[val_idx].astype(np.float32)
                    
                    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
                    val_data = lgb.Dataset(X_val, label=y_val, free_raw_data=True)
                    
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=200,
                        valid_sets=[val_data],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=20),
                            lgb.log_evaluation(0)
                        ]
                    )
                    
                    val_preds = model.predict(X_val)
                    brier_score = np.mean((val_preds - y_val) ** 2)
                    cv_scores.append(brier_score)
                    
                    # 清理内存
                    del train_data, val_data, model
                    gc.collect()
                
            return np.mean(cv_scores)
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        print("\n=== 参数优化完成 ===")
        print("最佳参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"最佳Brier Score: {study.best_value:.4f}")
        
        best_params = study.best_params
        best_params.update({
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "n_jobs": 1
        })
        
        print("\n使用最佳参数训练最终模型...")
        train_data = lgb.Dataset(self.train_features, label=self.train_labels)
        val_data = lgb.Dataset(self.val_features, label=self.val_labels)
        
        final_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)
            ]
        )
        
        return best_params, final_model 