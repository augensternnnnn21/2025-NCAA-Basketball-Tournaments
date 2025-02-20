import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
import time

class FeatureEngineering:
    """特征工程类，用于生成和处理模型所需的特征"""
    
    def __init__(self):
        """初始化特征工程类"""
        self.scaler = StandardScaler()
        # 扩展基础特征列
        self.base_cols = [
            "WinRate", "TotalGames", "Wins", "Losses", 
            "OffensiveRating", "DefensiveRating", "NetRating",
            "PaceAdjustedOffRating", "PaceAdjustedDefRating",
            "WinStreak", "PointsPerGame", "PointsAllowed",
            "FieldGoalPct", "ThreePointPct", "FreeThrowPct",
            "ReboundRate", "AssistRate", "TurnoverRate",
            "BlockRate", "StealRate", "PowerRating"
        ]
        
        # 更新特征列名列表
        self.feature_columns = [
            f"{col}_{suffix}" 
            for col in self.base_cols 
            for suffix in ["diff", "ratio", "trend"]
        ]
        self.feature_columns.extend([
            "historical_win_rate", "total_matchups",
            "recent_matchup_win_rate", "head_to_head_dominance",
            "seed_diff", "seed_ratio", "seed_historical_performance",
            "conference_strength_diff", "conference_win_rate_diff",
            "recent_win_rate_diff", "recent_win_rate_ratio",
            "momentum_score_diff", "momentum_score_ratio",
            "avg_score_margin_diff", "avg_score_margin_ratio",
            "quality_wins_diff", "upset_wins_diff",
            "tournament_experience_diff", "clutch_performance_diff",
            "strength_of_schedule_diff", "road_win_rate_diff",
            "neutral_court_win_rate_diff", "big_game_performance_diff"
        ])

    def calculate_team_power_rating(
        self,
        team_stats: pd.DataFrame,
        season_games: pd.DataFrame,
        team_id: int
    ) -> float:
        """计算球队综合实力评级"""
        # 基础胜率权重
        win_rate = team_stats["WinRate"].values[0] * 0.3
        
        # 对手实力加权胜率
        team_games = season_games[
            (season_games["WTeamID"] == team_id) |
            (season_games["LTeamID"] == team_id)
        ]
        opponent_win_rates = []
        for _, game in team_games.iterrows():
            opp_id = game["LTeamID"] if game["WTeamID"] == team_id else game["WTeamID"]
            opp_games = season_games[
                (season_games["WTeamID"] == opp_id) |
                (season_games["LTeamID"] == opp_id)
            ]
            opp_wins = len(opp_games[opp_games["WTeamID"] == opp_id])
            opp_win_rate = opp_wins / len(opp_games) if len(opp_games) > 0 else 0.5
            opponent_win_rates.append(opp_win_rate)
        
        strength_of_schedule = np.mean(opponent_win_rates) if opponent_win_rates else 0.5
        adjusted_win_rate = win_rate * strength_of_schedule * 0.3
        
        # 得分效率
        offensive_rating = team_stats["OffensiveRating"].values[0] / 100 * 0.2
        defensive_rating = (100 - team_stats["DefensiveRating"].values[0]) / 100 * 0.2
        
        return win_rate + adjusted_win_rate + offensive_rating + defensive_rating

    def create_matchup_features(
        self,
        team1_stats: pd.DataFrame,
        team2_stats: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        season_games: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """增强版对抗特征创建"""
        if feature_cols is None:
            feature_cols = self.base_cols
            
        # 确保所有需要的列都存在
        for col in feature_cols:
            if col not in team1_stats.columns or col not in team2_stats.columns:
                if col == "PowerRating" and season_games is not None:
                    # 计算球队实力评级
                    team1_stats.loc[:, "PowerRating"] = self.calculate_team_power_rating(
                        team1_stats, season_games, team1_stats["TeamID"].values[0]
                    )
                    team2_stats.loc[:, "PowerRating"] = self.calculate_team_power_rating(
                        team2_stats, season_games, team2_stats["TeamID"].values[0]
                    )
                else:
                    team1_stats.loc[:, col] = 0
                    team2_stats.loc[:, col] = 0
            
        matchup_features = []
        
        for col in feature_cols:
            # 直接差值
            matchup_features.append(
                team1_stats[col].values - team2_stats[col].values
            )
            # 比值
            matchup_features.append(
                team1_stats[col].values / (team2_stats[col].values + 1e-6)
            )
            # 趋势（如果有历史数据）
            if season_games is not None:
                team1_trend = self.calculate_trend(team1_stats, season_games, col)
                team2_trend = self.calculate_trend(team2_stats, season_games, col)
                matchup_features.append(team1_trend - team2_trend)
            else:
                matchup_features.append(0)
            
        matchup_df = pd.DataFrame(
            np.column_stack(matchup_features),
            columns=[f"{col}_{suffix}" 
                    for col in feature_cols 
                    for suffix in ["diff", "ratio", "trend"]]
        )
        
        # 初始化其他特征列为0
        for col in self.feature_columns:
            if col not in matchup_df.columns:
                matchup_df[col] = 0
                
        return matchup_df

    def calculate_trend(
        self,
        team_stats: pd.DataFrame,
        season_games: pd.DataFrame,
        metric: str,
        window: int = 5
    ) -> float:
        """计算指标的近期趋势"""
        team_id = team_stats["TeamID"].values[0]
        recent_games = season_games[
            (season_games["WTeamID"] == team_id) |
            (season_games["LTeamID"] == team_id)
        ].tail(window)
        
        if len(recent_games) < 2:
            return 0
            
        values = []
        for _, game in recent_games.iterrows():
            if metric in game:
                values.append(game[metric])
            elif metric == "WinRate":
                values.append(1 if game["WTeamID"] == team_id else 0)
                
        if len(values) < 2:
            return 0
            
        # 使用简单线性回归计算趋势
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def add_historical_matchup_features(
        self,
        matchup_df: pd.DataFrame,
        historical_games: pd.DataFrame,
        team1_id: int,
        team2_id: int,
        lookback_years: int = 5
    ) -> pd.DataFrame:
        """增强版历史交手特征"""
        current_season = historical_games["Season"].max()
        historical_matchups = historical_games[
            (historical_games["Season"] >= current_season - lookback_years) &
            (
                ((historical_games["WTeamID"] == team1_id) & 
                 (historical_games["LTeamID"] == team2_id)) |
                ((historical_games["WTeamID"] == team2_id) & 
                 (historical_games["LTeamID"] == team1_id))
            )
        ]
        
        if len(historical_matchups) == 0:
            matchup_df["historical_win_rate"] = 0.5
            matchup_df["total_matchups"] = 0
            matchup_df["recent_matchup_win_rate"] = 0.5
            matchup_df["head_to_head_dominance"] = 0
            return matchup_df
            
        # 计算历史交手统计
        team1_wins = historical_matchups[
            historical_games["WTeamID"] == team1_id
        ]
        total_games = len(historical_matchups)
        
        # 总体胜率
        win_rate = len(team1_wins) / total_games
        matchup_df["historical_win_rate"] = win_rate
        matchup_df["total_matchups"] = total_games
        
        # 近期交手胜率（最近3场）
        recent_matchups = historical_matchups.tail(3)
        recent_wins = len(recent_matchups[
            recent_matchups["WTeamID"] == team1_id
        ])
        matchup_df["recent_matchup_win_rate"] = (
            recent_wins / len(recent_matchups)
            if len(recent_matchups) > 0 else 0.5
        )
        
        # 计算支配度（考虑比分差距）
        score_diffs = []
        for _, game in historical_matchups.iterrows():
            if game["WTeamID"] == team1_id:
                score_diffs.append(game["WScore"] - game["LScore"])
            else:
                score_diffs.append(game["LScore"] - game["WScore"])
        
        matchup_df["head_to_head_dominance"] = np.mean(score_diffs)
        
        return matchup_df

    def add_tournament_seed_features(
        self,
        matchup_df: pd.DataFrame,
        tournament_seeds: pd.DataFrame,
        team1_id: int,
        team2_id: int,
        season: int
    ) -> pd.DataFrame:
        """增强版种子排名特征"""
        season_seeds = tournament_seeds[
            tournament_seeds["Season"] == season
        ]
        
        team1_seed = season_seeds[
            season_seeds["TeamID"] == team1_id
        ]["Seed"].values[0]
        team2_seed = season_seeds[
            season_seeds["TeamID"] == team2_id
        ]["Seed"].values[0]
        
        # 提取数字种子值
        team1_seed_num = int(team1_seed[1:3])
        team2_seed_num = int(team2_seed[1:3])
        
        # 基本种子差异
        matchup_df["seed_diff"] = team1_seed_num - team2_seed_num
        matchup_df["seed_ratio"] = team1_seed_num / team2_seed_num
        
        # 历史种子表现
        historical_seeds = tournament_seeds[
            tournament_seeds["Season"] < season
        ]
        
        def get_seed_performance(team_id):
            team_seeds = historical_seeds[
                historical_seeds["TeamID"] == team_id
            ]["Seed"].apply(lambda x: int(x[1:3]))
            return team_seeds.mean() if len(team_seeds) > 0 else 8
            
        team1_hist = get_seed_performance(team1_id)
        team2_hist = get_seed_performance(team2_id)
        matchup_df["seed_historical_performance"] = team1_hist - team2_hist
        
        return matchup_df

    def add_momentum_features(
        self,
        matchup_df: pd.DataFrame,
        recent_games: pd.DataFrame,
        team1_id: int,
        team2_id: int,
        n_games: int = 10
    ) -> pd.DataFrame:
        """增强版动量特征"""
        def get_team_momentum(team_id: int) -> Dict[str, float]:
            team_games = recent_games[
                (recent_games["WTeamID"] == team_id) |
                (recent_games["LTeamID"] == team_id)
            ].sort_values("DayNum", ascending=False).head(n_games)
            
            if len(team_games) == 0:
                return {
                    "recent_win_rate": 0.5,
                    "avg_score_margin": 0,
                    "momentum_score": 0,
                    "quality_wins": 0,
                    "upset_wins": 0,
                    "clutch_performance": 0.5
                }
                
            # 计算基本统计
            wins = team_games[team_games["WTeamID"] == team_id]
            win_rate = len(wins) / len(team_games)
            
            # 计算得分差
            margins = []
            quality_wins = 0
            upset_wins = 0
            clutch_games = 0
            clutch_wins = 0  # 新增变量
            
            for _, game in team_games.iterrows():
                is_win = game["WTeamID"] == team_id
                margin = game["WScore"] - game["LScore"]
                if not is_win:
                    margin = -margin
                margins.append(margin)
                
                # 优质胜利（击败强队）
                if is_win and game["LTeamID"] in top_teams:
                    quality_wins += 1
                    
                # 爆冷胜利
                if is_win and team_id not in top_teams and game["LTeamID"] in top_teams:
                    upset_wins += 1
                    
                # 关键比赛表现
                if abs(margin) <= 5:
                    clutch_games += 1
                    if is_win:
                        clutch_wins += 1  # 使用新变量
                        
            avg_margin = np.mean(margins)
            momentum_score = win_rate * 0.4 + (avg_margin / 20) * 0.3 + quality_wins * 0.3
            
            clutch_rate = clutch_wins / clutch_games if clutch_games > 0 else 0.5
                    
            return {
                "recent_win_rate": win_rate,
                "avg_score_margin": avg_margin,
                "momentum_score": momentum_score,
                "quality_wins": quality_wins,
                "upset_wins": upset_wins,
                "clutch_performance": clutch_rate
            }
            
        # 定义强队（可以根据胜率或其他指标动态确定）
        top_teams = set()  # 这里需要实现确定强队的逻辑
            
        team1_momentum = get_team_momentum(team1_id)
        team2_momentum = get_team_momentum(team2_id)
        
        for key in team1_momentum:
            matchup_df[f"{key}_diff"] = (
                team1_momentum[key] - team2_momentum[key]
            )
            if key != "avg_score_margin":  # 得分差不需要比率
                matchup_df[f"{key}_ratio"] = (
                    team1_momentum[key] / (team2_momentum[key] + 1e-6)
                )
            
        return matchup_df
        
    def scale_features(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        fit: bool = True
    ) -> np.ndarray:
        """
        标准化特征
        """
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        else:
            features_array = features
            
        if fit:
            return self.scaler.fit_transform(features_array)
        return self.scaler.transform(features_array)

    def calculate_team_stats(self, season_games, team_id):
        """计算球队的统计数据"""
        # 获取球队的比赛
        team_games = season_games[
            (season_games['WTeamID'] == team_id) | 
            (season_games['LTeamID'] == team_id)
        ].copy()
        
        # 重置索引
        team_games = team_games.reset_index(drop=True)
        
        # 创建一个新的 DataFrame，指定数据类型
        team_stats = pd.DataFrame(index=team_games.index)
        
        # 初始化所有统计列，使用float64类型
        stats_columns = {
            'PowerRating': 'float64',
            'WinRate': 'float64',
            'TotalGames': 'int64',
            'Wins': 'int64',
            'Losses': 'int64',
            'WinStreak': 'int64',
            'PointsPerGame': 'float64',
            'PointsAllowedPerGame': 'float64'
        }
        
        for col, dtype in stats_columns.items():
            team_stats[col] = pd.Series(0, index=team_games.index, dtype=dtype)
        
        if team_games.empty:
            return team_stats
        
        # 计算每场比赛的统计数据
        for i, game in team_games.iterrows():
            is_team1 = game['WTeamID'] == team_id
            team_score = game['WScore'] if is_team1 else game['LScore']
            opp_score = game['LScore'] if is_team1 else game['WScore']
            
            # 更新统计数据
            if i == 0:
                team_stats.at[i, 'TotalGames'] = 1
                team_stats.at[i, 'Wins'] = 1 if team_score > opp_score else 0
                team_stats.at[i, 'Losses'] = 1 if team_score < opp_score else 0
                team_stats.at[i, 'WinStreak'] = 1 if team_score > opp_score else 0
                team_stats.at[i, 'PointsPerGame'] = float(team_score)
                team_stats.at[i, 'PointsAllowedPerGame'] = float(opp_score)
            else:
                team_stats.at[i, 'TotalGames'] = team_stats.at[i-1, 'TotalGames'] + 1
                team_stats.at[i, 'Wins'] = team_stats.at[i-1, 'Wins'] + (1 if team_score > opp_score else 0)
                team_stats.at[i, 'Losses'] = team_stats.at[i-1, 'Losses'] + (1 if team_score < opp_score else 0)
                
                # 计算连胜
                prev_streak = team_stats.at[i-1, 'WinStreak']
                team_stats.at[i, 'WinStreak'] = prev_streak + 1 if team_score > opp_score else 0
                
                # 计算得分和失分
                team_stats.at[i, 'PointsPerGame'] = ((team_stats.at[i-1, 'PointsPerGame'] * i) + team_score) / (i + 1)
                team_stats.at[i, 'PointsAllowedPerGame'] = ((team_stats.at[i-1, 'PointsAllowedPerGame'] * i) + opp_score) / (i + 1)
            
            # 计算胜率
            team_stats.at[i, 'WinRate'] = team_stats.at[i, 'Wins'] / team_stats.at[i, 'TotalGames']
            
            # 计算 Power Rating
            team_stats.at[i, 'PowerRating'] = (team_stats.at[i, 'WinRate'] * 0.4 + 
                                              (team_stats.at[i, 'PointsPerGame'] - team_stats.at[i, 'PointsAllowedPerGame']) * 0.4 +
                                              team_stats.at[i, 'WinStreak'] * 0.2)
        
        return team_stats

    def calculate_historical_matchup_features(self, historical_matchups, team1_id, team2_id):
        # 创建一个新的 DataFrame
        matchup_stats = pd.DataFrame(index=[0])
        
        # 获取两队之间的历史交手记录
        team1_home = historical_matchups[
            (historical_matchups['WTeamID'] == team1_id) & 
            (historical_matchups['LTeamID'] == team2_id)
        ].copy()
        
        team1_away = historical_matchups[
            (historical_matchups['WTeamID'] == team2_id) & 
            (historical_matchups['LTeamID'] == team1_id)
        ].copy()
        
        # 计算历史交手胜率
        team1_home_wins = team1_home[team1_home['WScore'] > team1_home['LScore']].shape[0]
        team1_away_wins = team1_away[team1_away['LScore'] > team1_away['WScore']].shape[0]
        total_games = len(team1_home) + len(team1_away)
        
        if total_games > 0:
            matchup_stats.loc[0, 'HistoricalWinRate'] = (team1_home_wins + team1_away_wins) / total_games
            
            # 计算平均得分差
            home_score_diff = team1_home['WScore'].mean() - team1_home['LScore'].mean() if len(team1_home) > 0 else 0
            away_score_diff = team1_away['LScore'].mean() - team1_away['WScore'].mean() if len(team1_away) > 0 else 0
            matchup_stats.loc[0, 'AvgScoreDiff'] = (home_score_diff + away_score_diff) / 2 if total_games > 0 else 0
        else:
            matchup_stats.loc[0, 'HistoricalWinRate'] = 0.5
            matchup_stats.loc[0, 'AvgScoreDiff'] = 0
        
        return matchup_stats

    def process_data(self):
        """处理数据并生成特征"""
        print("\n=== 数据准备开始 ===")
        
        # 记录开始时间
        start_time = time.time()
        
        # 加载数据
        games_df = pd.concat([
            pd.read_csv('MRegularSeasonCompactResults.csv'),
            pd.read_csv('MNCAATourneyCompactResults.csv')
        ])
        teams_df = pd.read_csv('MTeams.csv')
        
        # 初始化结果列表
        features_list = []
        labels_list = []
        
        # 按赛季分组处理
        total_games = len(games_df)
        processed_games = 0
        
        # 初始化特征列
        feature_columns = [
            'PowerRating', 'WinRate', 'TotalGames', 'Wins', 'Losses',
            'WinStreak', 'PointsPerGame', 'PointsAllowedPerGame',
            'HistoricalWinRate', 'AvgScoreDiff'
        ]
        
        for season in games_df['Season'].unique():
            season_games = games_df[games_df['Season'] == season].copy()
            season_games = season_games.sort_values('DayNum').reset_index(drop=True)
            
            # 处理每场比赛
            for i, game in season_games.iterrows():
                # 获取胜队和负队的ID
                win_team_id = game['WTeamID']
                lose_team_id = game['LTeamID']
                
                # 为胜队生成特征（作为team1）
                team1_stats = self.calculate_team_stats(
                    season_games.iloc[:i],
                    win_team_id
                )
                team2_stats = self.calculate_team_stats(
                    season_games.iloc[:i],
                    lose_team_id
                )
                
                # 获取历史交手数据
                historical_matchups = season_games.iloc[:i].copy()
                
                # 生成胜队作为team1的特征
                win_features = self._create_feature_vector(
                    team1_stats, team2_stats,
                    historical_matchups,
                    win_team_id, lose_team_id,
                    feature_columns
                )
                features_list.append(win_features)
                labels_list.append(1)  # 胜队标签为1
                
                # 生成负队作为team1的特征
                lose_features = self._create_feature_vector(
                    team2_stats, team1_stats,
                    historical_matchups,
                    lose_team_id, win_team_id,
                    feature_columns
                )
                features_list.append(lose_features)
                labels_list.append(0)  # 负队标签为0
                
                # 更新进度
                processed_games += 1
                if processed_games % 10 == 0:
                    speed = processed_games / (time.time() - start_time)
                    remaining_time = (total_games - processed_games) / speed
                    print(f"\r   已处理 {processed_games}/{total_games} 场比赛 "
                          f"(速度: {speed:.1f} 场/秒, 预计还需 {remaining_time:.1f} 秒)",
                          end='', flush=True)
        
        print(f"\n\n=== 数据准备完成 ===")
        print(f"总共处理了 {total_games} 场比赛")
        print(f"收集了 {len(features_list)} 个样本\n")
        
        # 转换为 numpy 数组
        features_array = np.array(features_list, dtype=np.float32)
        labels_array = np.array(labels_list, dtype=np.float32)
        
        # 打印特征维度信息
        print(f"特征维度: {features_array.shape}")
        print(f"标签维度: {labels_array.shape}")
        print("\n特征列说明:")
        feature_names = (
            [f"team1_{col}" for col in feature_columns[:8]] +  # 球队1统计
            [f"team2_{col}" for col in feature_columns[:8]] +  # 球队2统计
            feature_columns[8:] +  # 历史交手特征
            [f"{col}_diff" for col in feature_columns[:8]] +  # 差值特征
            [f"{col}_ratio" for col in feature_columns[:8]]   # 比值特征
        )
        for i, name in enumerate(feature_names):
            print(f"{i+1:3d}. {name}")
        
        return features_array, labels_array

    def _create_feature_vector(
        self,
        team1_stats: pd.DataFrame,
        team2_stats: pd.DataFrame,
        historical_matchups: pd.DataFrame,
        team1_id: int,
        team2_id: int,
        feature_columns: List[str]
    ) -> List[float]:
        """创建特征向量"""
        matchup_features = self.calculate_historical_matchup_features(
            historical_matchups,
            team1_id,
            team2_id
        )
        
        feature_vector = []
        
        # 添加球队1的统计数据
        if not team1_stats.empty:
            for col in feature_columns[:8]:  # 前8列是球队统计数据
                feature_vector.append(float(team1_stats.iloc[-1][col]))
        else:
            feature_vector.extend([0.0] * 8)
            
        # 添加球队2的统计数据
        if not team2_stats.empty:
            for col in feature_columns[:8]:
                feature_vector.append(float(team2_stats.iloc[-1][col]))
        else:
            feature_vector.extend([0.0] * 8)
            
        # 添加历史交手特征
        feature_vector.append(float(matchup_features.iloc[0]['HistoricalWinRate']))
        feature_vector.append(float(matchup_features.iloc[0]['AvgScoreDiff']))
        
        # 添加差值特征
        if not team1_stats.empty and not team2_stats.empty:
            for col in feature_columns[:8]:
                diff = float(team1_stats.iloc[-1][col]) - float(team2_stats.iloc[-1][col])
                ratio = float(team1_stats.iloc[-1][col]) / (float(team2_stats.iloc[-1][col]) + 1e-6)
                feature_vector.extend([diff, ratio])
        else:
            feature_vector.extend([0.0] * 16)  # 8个特征的差值和比值
        
        return feature_vector 