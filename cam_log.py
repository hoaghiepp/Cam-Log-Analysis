import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_decision_scores(csv_file_path):
    """
    Phân tích decision scores: best_score, score_max, mahalanobis, norm_entropy
    """
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Decision score columns
    score_cols = ['best_score', 'score_max', 'mahalanobis', 'norm_entropy']
    available_cols = [col for col in score_cols if col in df.columns]
    
    print(f"📊 Decision Scores Analysis - {len(df):,} records")
    print(f"Available columns: {available_cols}")
    
    return df, available_cols

def decision_scores_summary(df, score_cols):
    """
    Statistical summary của decision scores
    """
    print("\n" + "="*50)
    print("DECISION SCORES STATISTICAL SUMMARY")
    print("="*50)
    
    summary_stats = df[score_cols].describe().round(4)
    print(summary_stats)
    
    # Key insights
    print("\n📋 KEY INSIGHTS:")
    for col in score_cols:
        data = df[col].dropna()
        if len(data) > 0:
            q95 = np.percentile(data, 95)
            q05 = np.percentile(data, 5)
            print(f"{col}:")
            print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  95% data trong: [{q05:.4f}, {q95:.4f}]")
            print(f"  Outliers (>P95): {len(data[data > q95])} ({len(data[data > q95])/len(data)*100:.1f}%)")
    
    return summary_stats

def plot_decision_score_distributions(df, score_cols):
    """
    Vẽ distribution plots cho decision scores
    """
    n_cols = len(score_cols)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
    
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    
    for i, col in enumerate(score_cols):
        data = df[col].dropna()
        
        # Histogram
        axes[0, i].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.3f}')
        axes[0, i].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.3f}')
        axes[0, i].set_title(f'{col} Distribution')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, i].boxplot(data, vert=True)
        axes[1, i].set_title(f'{col} Box Plot')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add outlier info
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
        axes[1, i].text(1.1, Q3, f'Outliers: {len(outliers)}', transform=axes[1, i].get_yaxis_transform())
    
    plt.tight_layout()
    plt.show()

def analyze_score_correlations(df, score_cols):
    """
    Phân tích correlation giữa decision scores
    """
    print("\n" + "="*50)
    print("DECISION SCORES CORRELATION ANALYSIS")
    print("="*50)
    
    corr_matrix = df[score_cols].corr()
    print(corr_matrix.round(3))
    
    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={"shrink": .8})
    plt.title('Decision Scores Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Key correlation insights
    print("\n🔍 CORRELATION INSIGHTS:")
    for i, col1 in enumerate(score_cols):
        for j, col2 in enumerate(score_cols[i+1:], i+1):
            corr_val = corr_matrix.loc[col1, col2]
            if abs(corr_val) > 0.5:
                relationship = "Strong positive" if corr_val > 0.5 else "Strong negative"
                print(f"  {col1} vs {col2}: {relationship} correlation ({corr_val:.3f})")
    
    return corr_matrix

def detect_decision_anomalies(df, score_cols, z_threshold=3.0):
    """
    Phát hiện anomalies trong decision scores
    """
    print(f"\n🚨 ANOMALY DETECTION (Z-score > {z_threshold})")
    print("="*50)
    
    anomalies = {}
    total_anomalies = 0
    
    for col in score_cols:
        data = df[col].dropna()
        if len(data) > 0:
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > z_threshold]
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
            
            anomalies[col] = {
                'z_score_outliers': len(outliers),
                'iqr_outliers': len(iqr_outliers),
                'z_score_threshold': data.mean() + z_threshold * data.std(),
                'extreme_values': outliers.tolist()[:5]  # Top 5 extreme values
            }
            
            total_anomalies += len(outliers)
            
            print(f"{col}:")
            print(f"  Z-score outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
            print(f"  IQR outliers: {len(iqr_outliers)} ({len(iqr_outliers)/len(data)*100:.1f}%)")
            if len(outliers) > 0:
                print(f"  Most extreme: {outliers.nlargest(3).values}")
    
    print(f"\n📊 Total anomalous records: {total_anomalies}")
    return anomalies

def build_anomaly_scenarios(df, score_cols):
    """
    Xây dựng kịch bản phát hiện bất thường cho decision scores
    """
    print("\n" + "="*60)
    print("🎯 ANOMALY DETECTION SCENARIOS")
    print("="*60)
    
    scenarios = []
    
    # Scenario 1: Very low best_score (failed recognition)
    if 'best_score' in score_cols:
        low_best_score = df[df['best_score'] < 0.3]
        scenarios.append({
            'name': 'Low Recognition Confidence',
            'condition': 'best_score < 0.3',
            'count': len(low_best_score),
            'percentage': len(low_best_score)/len(df)*100,
            'impact': 'Potential false positives or poor face quality',
            'action': 'Review image quality, adjust thresholds'
        })
    
    # Scenario 2: High mahalanobis distance (unknown faces)
    if 'mahalanobis' in score_cols:
        high_mahalanobis = df[df['mahalanobis'] > df['mahalanobis'].quantile(0.95)]
        scenarios.append({
            'name': 'Unknown Face Detection',
            'condition': f'mahalanobis > P95 ({df["mahalanobis"].quantile(0.95):.2f})',
            'count': len(high_mahalanobis),
            'percentage': len(high_mahalanobis)/len(df)*100,
            'impact': 'New faces or faces very different from known database',
            'action': 'Review for new person enrollment'
        })
    
    # Scenario 3: Low entropy (uncertain decisions)
    if 'norm_entropy' in score_cols:
        low_entropy = df[df['norm_entropy'] < df['norm_entropy'].quantile(0.1)]
        scenarios.append({
            'name': 'Low Decision Entropy',
            'condition': f'norm_entropy < P10 ({df["norm_entropy"].quantile(0.1):.2f})',
            'count': len(low_entropy),
            'percentage': len(low_entropy)/len(df)*100,
            'impact': 'Very certain decisions (good or concerning)',
            'action': 'Monitor for potential bias or overfitting'
        })
    
    # Scenario 4: Inconsistent scores (best_score vs score_max gap)
    if 'best_score' in score_cols and 'score_max' in score_cols:
        df['score_gap'] = df['score_max'] - df['best_score']
        large_gap = df[df['score_gap'] > df['score_gap'].quantile(0.9)]
        scenarios.append({
            'name': 'Inconsistent Score Gap',
            'condition': f'score_max - best_score > P90 ({df["score_gap"].quantile(0.9):.2f})',
            'count': len(large_gap),
            'percentage': len(large_gap)/len(df)*100,
            'impact': 'Inconsistent similarity calculations',
            'action': 'Check algorithm consistency, potential model issues'
        })
    
    # Scenario 5: Multi-condition anomalies
    if len(score_cols) >= 2:
        # Combine multiple conditions
        multi_anomaly = df[
            (df[score_cols[0]] > df[score_cols[0]].quantile(0.95)) &
            (df[score_cols[1]] < df[score_cols[1]].quantile(0.05))
        ]
        scenarios.append({
            'name': 'Multi-Score Anomaly',
            'condition': f'{score_cols[0]} > P95 AND {score_cols[1]} < P05',
            'count': len(multi_anomaly),
            'percentage': len(multi_anomaly)/len(df)*100,
            'impact': 'Complex anomalous behavior requiring investigation',
            'action': 'Deep dive analysis, potential system malfunction'
        })
    
    # Print scenarios
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print(f"   Condition: {scenario['condition']}")
        print(f"   Occurrences: {scenario['count']} ({scenario['percentage']:.1f}%)")
        print(f"   Impact: {scenario['impact']}")
        print(f"   Action: {scenario['action']}")
    
    return scenarios

def create_anomaly_alerts(df, score_cols):
    """
    Tạo thresholds cho real-time anomaly alerts
    """
    print("\n" + "="*60)
    print("🚨 REAL-TIME ALERT THRESHOLDS")
    print("="*60)
    
    alert_thresholds = {}
    
    for col in score_cols:
        data = df[col].dropna()
        if len(data) > 0:
            # Calculate thresholds
            mean_val = data.mean()
            std_val = data.std()
            
            alert_thresholds[col] = {
                'warning_low': np.percentile(data, 5),
                'warning_high': np.percentile(data, 95),
                'critical_low': np.percentile(data, 1),
                'critical_high': np.percentile(data, 99),
                'z_score_critical': 3.0,
                'current_mean': mean_val,
                'current_std': std_val
            }
            
            print(f"\n{col} Alert Thresholds:")
            print(f"  Warning: [{alert_thresholds[col]['warning_low']:.4f}, {alert_thresholds[col]['warning_high']:.4f}]")
            print(f"  Critical: [{alert_thresholds[col]['critical_low']:.4f}, {alert_thresholds[col]['critical_high']:.4f}]")
    
    # Generate alert code template
    print(f"\n💻 ALERT CODE TEMPLATE:")
    print("""
def check_decision_score_alerts(new_scores):
    alerts = []
    thresholds = alert_thresholds  # Use calculated thresholds
    
    for col, value in new_scores.items():
        if col in thresholds:
            t = thresholds[col]
            if value < t['critical_low'] or value > t['critical_high']:
                alerts.append(f"CRITICAL: {col} = {value:.4f}")
            elif value < t['warning_low'] or value > t['warning_high']:
                alerts.append(f"WARNING: {col} = {value:.4f}")
    
    return alerts
    """)
    
    return alert_thresholds

def main_decision_analysis(csv_file_path):
    """
    Main function thực hiện complete decision scores analysis
    """
    # Load and prepare data
    df, score_cols = analyze_decision_scores(csv_file_path)
    
    if not score_cols:
        print("❌ No decision score columns found")
        return None
    
    # Statistical summary
    summary_stats = decision_scores_summary(df, score_cols)
    
    # Distribution plots
    plot_decision_score_distributions(df, score_cols)
    
    # Correlation analysis
    corr_matrix = analyze_score_correlations(df, score_cols)
    
    # Anomaly detection
    anomalies = detect_decision_anomalies(df, score_cols)
    
    # Build anomaly scenarios
    scenarios = build_anomaly_scenarios(df, score_cols)
    
    # Create alert thresholds
    alert_thresholds = create_anomaly_alerts(df, score_cols)
    
    print("\n" + "="*60)
    print("✅ DECISION SCORES ANALYSIS COMPLETED")
    print("="*60)
    
    return {
        'summary_stats': summary_stats,
        'correlations': corr_matrix,
        'anomalies': anomalies,
        'scenarios': scenarios,
        'alert_thresholds': alert_thresholds,
        'data': df
    }

# Usage example:
results = main_decision_analysis(r'data/face_decision_events_last_7_days_20250905_110624.csv')