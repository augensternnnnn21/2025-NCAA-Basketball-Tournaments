import pandas as pd
import numpy as np
from pathlib import Path

class MarchMadnessDataLoader:
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.regular_season_results = None
        self.tourney_results = None
        self.team_data = None
        self.seasons = None
        self.massey_ordinals = None
        
    def load_regular_season_results(self, detailed: bool = True):
        """加载常规赛结果"""
        if detailed:
            self.regular_season_results = pd.read_csv(
                self.data_dir / "MRegularSeasonDetailedResults.csv"
            )
        else:
            self.regular_season_results = pd.read_csv(
                self.data_dir / "MRegularSeasonCompactResults.csv"
            )
        return self.regular_season_results
    
    def load_tournament_results(self, detailed: bool = True):
        """加载锦标赛结果"""
        if detailed:
            self.tourney_results = pd.read_csv(
                self.data_dir / "MNCAATourneyDetailedResults.csv"
            )
        else:
            self.tourney_results = pd.read_csv(
                self.data_dir / "MNCAATourneyCompactResults.csv"
            )
        return self.tourney_results
    
    def load_team_data(self):
        """加载球队基础数据"""
        self.team_data = pd.read_csv(self.data_dir / "MTeams.csv")
        return self.team_data
    
    def load_seasons(self):
        """加载赛季信息"""
        self.seasons = pd.read_csv(self.data_dir / "MSeasons.csv")
        return self.seasons
    
    def load_massey_ordinals(self):
        """加载Massey排名数据"""
        self.massey_ordinals = pd.read_csv(self.data_dir / "MMasseyOrdinals.csv")
        return self.massey_ordinals

    def load_tournament_seeds(self):
        """加载锦标赛种子排名数据"""
        return pd.read_csv(self.data_dir / "MNCAATourneySeeds.csv")
    
    def calculate_basic_team_stats(self, season: int):
        """计算球队基础统计数据"""
        if self.regular_season_results is None:
            self.load_regular_season_results()
            
        if self.team_data is None:
            self.load_team_data()
            
        season_games = self.regular_season_results[
            self.regular_season_results["Season"] == season
        ]
        
        team_stats = []
        for team_id in self.team_data["TeamID"].unique():
            # 获取球队比赛
            team_wins = season_games[season_games["WTeamID"] == team_id]
            team_losses = season_games[season_games["LTeamID"] == team_id]
            total_games = len(team_wins) + len(team_losses)
            
            if total_games == 0:
                continue
                
            # 基本统计
            wins = len(team_wins)
            losses = len(team_losses)
            win_rate = wins / total_games if total_games > 0 else 0
            
            # 得分统计
            points_scored = team_wins["WScore"].sum() + team_losses["LScore"].sum()
            points_allowed = team_wins["LScore"].sum() + team_losses["WScore"].sum()
            points_per_game = points_scored / total_games
            points_allowed_per_game = points_allowed / total_games
            
            # 计算连胜
            games_chronological = pd.concat([
                team_wins[["Season", "DayNum", "WTeamID"]].assign(won=1),
                team_losses[["Season", "DayNum", "LTeamID"]].assign(won=0)
            ]).sort_values("DayNum")
            current_streak = 0
            for won in games_chronological["won"]:
                if won:
                    current_streak = max(0, current_streak + 1)
                else:
                    current_streak = min(0, current_streak - 1)
            
            # 计算投篮命中率
            fg_attempts = (
                team_wins["WFGA"].sum() + team_losses["LFGA"].sum()
            )
            fg_made = (
                team_wins["WFGM"].sum() + team_losses["LFGM"].sum()
            )
            fg_pct = fg_made / fg_attempts if fg_attempts > 0 else 0
            
            three_attempts = (
                team_wins["WFGA3"].sum() + team_losses["LFGA3"].sum()
            )
            three_made = (
                team_wins["WFGM3"].sum() + team_losses["LFGM3"].sum()
            )
            three_pct = three_made / three_attempts if three_attempts > 0 else 0
            
            ft_attempts = (
                team_wins["WFTA"].sum() + team_losses["LFTA"].sum()
            )
            ft_made = (
                team_wins["WFTM"].sum() + team_losses["LFTM"].sum()
            )
            ft_pct = ft_made / ft_attempts if ft_attempts > 0 else 0
            
            # 计算进阶统计
            total_possessions = (
                team_wins["WFGA"].sum() - team_wins["WOR"].sum() + 
                team_wins["WTO"].sum() + 0.44 * team_wins["WFTA"].sum() +
                team_losses["LFGA"].sum() - team_losses["LOR"].sum() + 
                team_losses["LTO"].sum() + 0.44 * team_losses["LFTA"].sum()
            )
            
            offensive_rating = 100 * points_scored / total_possessions if total_possessions > 0 else 0
            defensive_rating = 100 * points_allowed / total_possessions if total_possessions > 0 else 0
            net_rating = offensive_rating - defensive_rating
            
            # 计算节奏
            pace = 40 * total_possessions / (total_games * 40)  # 假设每场比赛40分钟
            pace_adjusted_off = offensive_rating * (pace / 70)  # 标准化到70回合
            pace_adjusted_def = defensive_rating * (70 / pace)
            
            # 计算其他高级数据
            total_rebounds = (
                team_wins["WOR"].sum() + team_wins["WDR"].sum() +
                team_losses["LOR"].sum() + team_losses["LDR"].sum()
            )
            total_opp_rebounds = (
                team_wins["LOR"].sum() + team_wins["LDR"].sum() +
                team_losses["WOR"].sum() + team_losses["WDR"].sum()
            )
            rebound_rate = total_rebounds / (total_rebounds + total_opp_rebounds)
            
            total_assists = team_wins["WAst"].sum() + team_losses["LAst"].sum()
            total_fga = team_wins["WFGA"].sum() + team_losses["LFGA"].sum()
            assist_rate = total_assists / total_fga if total_fga > 0 else 0
            
            total_turnovers = team_wins["WTO"].sum() + team_losses["LTO"].sum()
            turnover_rate = total_turnovers / total_possessions if total_possessions > 0 else 0
            
            total_blocks = team_wins["WBlk"].sum() + team_losses["LBlk"].sum()
            total_opp_fga = (
                team_wins["LFGA"].sum() + team_losses["WFGA"].sum()
            )
            block_rate = total_blocks / total_opp_fga if total_opp_fga > 0 else 0
            
            total_steals = team_wins["WStl"].sum() + team_losses["LStl"].sum()
            steal_rate = total_steals / total_possessions if total_possessions > 0 else 0
            
            team_stats.append({
                "Season": season,
                "TeamID": team_id,
                "WinRate": win_rate,
                "TotalGames": total_games,
                "Wins": wins,
                "Losses": losses,
                "WinStreak": current_streak,
                "PointsPerGame": points_per_game,
                "PointsAllowed": points_allowed_per_game,
                "OffensiveRating": offensive_rating,
                "DefensiveRating": defensive_rating,
                "NetRating": net_rating,
                "PaceAdjustedOffRating": pace_adjusted_off,
                "PaceAdjustedDefRating": pace_adjusted_def,
                "FieldGoalPct": fg_pct,
                "ThreePointPct": three_pct,
                "FreeThrowPct": ft_pct,
                "ReboundRate": rebound_rate,
                "AssistRate": assist_rate,
                "TurnoverRate": turnover_rate,
                "BlockRate": block_rate,
                "StealRate": steal_rate
            })
            
        return pd.DataFrame(team_stats)
    
    def calculate_four_factors(self, season: int):
        """计算四因素(进攻效率、篮板率、失误率、罚球率)"""
        if self.regular_season_results is None:
            self.load_regular_season_results(detailed=True)
            
        if self.team_data is None:
            self.load_team_data()
            
        season_games = self.regular_season_results[
            self.regular_season_results["Season"] == season
        ]
        
        team_stats = []
        for team_id in self.team_data["TeamID"].unique():
            # 获取球队比赛
            team_wins = season_games[season_games["WTeamID"] == team_id]
            team_losses = season_games[season_games["LTeamID"] == team_id]
            
            total_games = len(team_wins) + len(team_losses)
            if total_games == 0:
                continue
                
            # 计算进攻回合
            win_possessions = (
                team_wins["WFGA"] - team_wins["WOR"] + 
                team_wins["WTO"] + 0.44 * team_wins["WFTA"]
            )
            loss_possessions = (
                team_losses["LFGA"] - team_losses["LOR"] + 
                team_losses["LTO"] + 0.44 * team_losses["LFTA"]
            )
            total_possessions = sum(win_possessions) + sum(loss_possessions)
            
            # 计算得分
            win_points = team_wins["WScore"].sum()
            loss_points = team_losses["LScore"].sum()
            total_points = win_points + loss_points
            
            # 计算进攻效率
            offensive_rating = (
                100 * total_points / total_possessions 
                if total_possessions > 0 else 0
            )
            
            # 计算防守效率
            opp_win_points = team_wins["LScore"].sum()
            opp_loss_points = team_losses["WScore"].sum()
            defensive_rating = (
                100 * (opp_win_points + opp_loss_points) / total_possessions
                if total_possessions > 0 else 0
            )
            
            # 计算净效率
            net_rating = offensive_rating - defensive_rating
            
            # 计算其他高级数据
            total_rebounds = (
                team_wins["WOR"].sum() + team_wins["WDR"].sum() +
                team_losses["LOR"].sum() + team_losses["LDR"].sum()
            )
            total_opp_rebounds = (
                team_wins["LOR"].sum() + team_wins["LDR"].sum() +
                team_losses["WOR"].sum() + team_losses["WDR"].sum()
            )
            rebound_rate = (
                total_rebounds / (total_rebounds + total_opp_rebounds)
                if (total_rebounds + total_opp_rebounds) > 0 else 0
            )
            
            # 计算失误率
            total_turnovers = (
                team_wins["WTO"].sum() + team_losses["LTO"].sum()
            )
            turnover_rate = (
                total_turnovers / total_possessions
                if total_possessions > 0 else 0
            )
            
            # 计算罚球率
            total_fta = (
                team_wins["WFTA"].sum() + team_losses["LFTA"].sum()
            )
            total_fga = (
                team_wins["WFGA"].sum() + team_losses["LFGA"].sum()
            )
            free_throw_rate = (
                total_fta / total_fga
                if total_fga > 0 else 0
            )
            
            team_stats.append({
                "Season": season,
                "TeamID": team_id,
                "OffensiveRating": offensive_rating,
                "DefensiveRating": defensive_rating,
                "NetRating": net_rating,
                "ReboundRate": rebound_rate,
                "TurnoverRate": turnover_rate,
                "FreeThrowRate": free_throw_rate
            })
            
        return pd.DataFrame(team_stats)
    
    def load_season_features(self, season: int):
        """准备锦标赛预测特征"""
        # 加载基础统计数据
        basic_stats = self.calculate_basic_team_stats(season)
        
        # 加载四因素数据
        four_factors = self.calculate_four_factors(season)
        
        # 合并特征
        features = pd.merge(
            basic_stats,
            four_factors,
            on=["Season", "TeamID"],
            suffixes=("", "_4factors")
        )
        
        # 处理重复列
        for col in features.columns:
            if col.endswith("_4factors"):
                base_col = col[:-9]
                if base_col in features.columns:
                    # 使用四因素的值替换基础统计值
                    features[base_col] = features[col]
                features.drop(col, axis=1, inplace=True)
        
        return features 