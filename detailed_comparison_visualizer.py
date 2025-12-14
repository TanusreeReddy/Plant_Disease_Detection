import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the comparison results
df = pd.read_csv('model_comparison/model_comparison_results.csv')

print("="*80)
print("DETAILED MODEL COMPARISON ANALYSIS")
print("="*80)

# Clean data
df['Train Acc (%)'] = df['Train Accuracy'].str.rstrip('%').astype(float)
df['Val Acc (%)'] = df['Val Accuracy'].str.rstrip('%').astype(float)
df['Test Acc (%)'] = df['Test Accuracy'].str.rstrip('%').astype(float)
df['Loss'] = df['Test Loss'].astype(float)
df['Params (M)'] = df['Parameters'].str.replace(',', '').astype(float) / 1_000_000
df['Time (min)'] = df['Training Time (sec)'].astype(float) / 60

print("\n📊 PROCESSED DATA:")
print(df[['Model', 'Train Acc (%)', 'Val Acc (%)', 'Test Acc (%)', 'Time (min)', 'Params (M)']].to_string(index=False))

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color scheme
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
model_colors = dict(zip(df['Model'], colors))

# 1. Test Accuracy Comparison (Large)
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.barh(df['Model'], df['Test Acc (%)'], color=colors, edgecolor='black', linewidth=2)
ax1.set_xlabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Test Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlim(0, 105)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels and performance indicators
for i, (bar, acc, model) in enumerate(zip(bars, df['Test Acc (%)'], df['Model'])):
    width = bar.get_width()
    label = f'{acc:.2f}%'
    
    # Performance indicator
    if acc >= 95:
        indicator = ' ⭐⭐⭐ Excellent'
        color = 'green'
    elif acc >= 85:
        indicator = ' ⭐⭐ Good'
        color = 'orange'
    elif acc >= 70:
        indicator = ' ⭐ Fair'
        color = 'darkorange'
    else:
        indicator = ' ⚠️ Poor'
        color = 'red'
    
    ax1.text(width + 2, i, label + indicator, va='center', fontweight='bold', fontsize=11, color=color)

# Add reference lines
ax1.axvline(x=95, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent (95%)')
ax1.axvline(x=85, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Good (85%)')
ax1.legend(loc='lower right', fontsize=10)

# 2. Training Time vs Accuracy Scatter
ax2 = fig.add_subplot(gs[0, 2])
scatter = ax2.scatter(df['Time (min)'], df['Test Acc (%)'], 
                     s=df['Params (M)']*20, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
for i, model in enumerate(df['Model']):
    ax2.annotate(model, (df['Time (min)'].iloc[i], df['Test Acc (%)'].iloc[i]), 
                fontsize=9, fontweight='bold', xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('Training Time (minutes)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Efficiency Analysis\n(Bubble size = Parameters)', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Add ideal zone
ax2.axhspan(95, 100, alpha=0.1, color='green', label='High Accuracy Zone')
ax2.axvspan(0, df['Time (min)'].median(), alpha=0.1, color='blue', label='Fast Training Zone')
ax2.legend(fontsize=8, loc='lower right')

# 3. Train vs Val vs Test Accuracy
ax3 = fig.add_subplot(gs[1, :])
x = np.arange(len(df['Model']))
width = 0.25

bars1 = ax3.bar(x - width, df['Train Acc (%)'], width, label='Training', color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax3.bar(x, df['Val Acc (%)'], width, label='Validation', color='#2ecc71', alpha=0.8, edgecolor='black')
bars3 = ax3.bar(x + width, df['Test Acc (%)'], width, label='Test', color='#e74c3c', alpha=0.8, edgecolor='black')

ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Training vs Validation vs Test Accuracy\n(Gap indicates overfitting)', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(df['Model'], fontsize=11)
ax3.legend(fontsize=11, loc='upper left')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 105)

# Add overfitting indicators
for i in range(len(df)):
    train_val_gap = df['Train Acc (%)'].iloc[i] - df['Val Acc (%)'].iloc[i]
    if train_val_gap > 5:
        ax3.annotate('⚠️ Overfitting', xy=(i, df['Train Acc (%)'].iloc[i]), 
                    xytext=(0, 10), textcoords='offset points',
                    fontsize=9, color='red', fontweight='bold',
                    ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# 4. Model Size (Parameters)
ax4 = fig.add_subplot(gs[2, 0])
bars = ax4.bar(df['Model'], df['Params (M)'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax4.set_ylabel('Parameters (Millions)', fontsize=11, fontweight='bold')
ax4.set_title('Model Size Comparison', fontsize=12, fontweight='bold', pad=15)
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3)

for bar, params in zip(bars, df['Params (M)']):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{params:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 5. Training Time Comparison
ax5 = fig.add_subplot(gs[2, 1])
bars = ax5.bar(df['Model'], df['Time (min)'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax5.set_ylabel('Training Time (minutes)', fontsize=11, fontweight='bold')
ax5.set_title('Training Time Comparison', fontsize=12, fontweight='bold', pad=15)
ax5.tick_params(axis='x', rotation=45)
ax5.grid(axis='y', alpha=0.3)

for bar, time in zip(bars, df['Time (min)']):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + max(df['Time (min)'])*0.02,
            f'{time:.0f}m', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 6. Efficiency Score (Custom Metric)
ax6 = fig.add_subplot(gs[2, 2])
# Efficiency = (Accuracy / Training_Time) * 100
efficiency = (df['Test Acc (%)'] / df['Time (min)']) * 10
bars = ax6.barh(df['Model'], efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax6.set_xlabel('Efficiency Score\n(Accuracy/Time)', fontsize=11, fontweight='bold')
ax6.set_title('Overall Efficiency Ranking', fontsize=12, fontweight='bold', pad=15)
ax6.grid(axis='x', alpha=0.3)

for bar, eff in zip(bars, efficiency):
    width = bar.get_width()
    ax6.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
            f'{eff:.2f}', va='center', fontweight='bold', fontsize=10)

# Add main title
fig.suptitle('Comprehensive CNN Model Comparison for Plant Disease Detection', 
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('model_comparison/detailed_comparison_visualization.png', dpi=200, bbox_inches='tight')
print("\n✓ Detailed visualization saved: model_comparison/detailed_comparison_visualization.png")

# Create Performance Summary Report
print("\n" + "="*80)
print("DETAILED PERFORMANCE ANALYSIS")
print("="*80)

for i, row in df.iterrows():
    print(f"\n{'='*80}")
    print(f"MODEL: {row['Model']}")
    print(f"{'='*80}")
    
    # Accuracy Analysis
    print(f"\n📊 ACCURACY METRICS:")
    print(f"  • Training Accuracy:   {row['Train Acc (%)']:.2f}%")
    print(f"  • Validation Accuracy: {row['Val Acc (%)']:.2f}%")
    print(f"  • Test Accuracy:       {row['Test Acc (%)']:.2f}% ", end="")
    
    if row['Test Acc (%)'] >= 95:
        print("⭐⭐⭐ EXCELLENT")
    elif row['Test Acc (%)'] >= 85:
        print("⭐⭐ GOOD")
    elif row['Test Acc (%)'] >= 70:
        print("⭐ FAIR")
    else:
        print("⚠️ POOR")
    
    # Overfitting Check
    train_val_gap = row['Train Acc (%)'] - row['Val Acc (%)']
    print(f"\n🎯 GENERALIZATION:")
    print(f"  • Train-Val Gap: {train_val_gap:.2f}%", end=" ")
    if train_val_gap > 5:
        print("⚠️ HIGH OVERFITTING RISK")
    elif train_val_gap > 2:
        print("⚡ MODERATE OVERFITTING")
    else:
        print("✓ GOOD GENERALIZATION")
    
    val_test_gap = abs(row['Val Acc (%)'] - row['Test Acc (%)'])
    print(f"  • Val-Test Gap:  {val_test_gap:.2f}%", end=" ")
    if val_test_gap < 2:
        print("✓ CONSISTENT PERFORMANCE")
    else:
        print("⚠️ INCONSISTENT")
    
    # Resource Usage
    print(f"\n⚙️ RESOURCE USAGE:")
    print(f"  • Parameters:     {row['Params (M)']:.2f}M", end=" ")
    if row['Params (M)'] < 5:
        print("(Lightweight)")
    elif row['Params (M)'] < 15:
        print("(Medium)")
    else:
        print("(Heavy)")
    
    print(f"  • Training Time:  {row['Time (min)']:.1f} minutes", end=" ")
    if row['Time (min)'] < 300:
        print("(Fast)")
    elif row['Time (min)'] < 600:
        print("(Moderate)")
    else:
        print("(Slow)")
    
    # Efficiency Score
    eff = (row['Test Acc (%)'] / row['Time (min)']) * 10
    print(f"  • Efficiency:     {eff:.2f}", end=" ")
    if eff > 0.5:
        print("(Excellent)")
    elif eff > 0.3:
        print("(Good)")
    else:
        print("(Poor)")
    
    # Use Case Recommendation
    print(f"\n💡 BEST FOR:")
    if row['Test Acc (%)'] >= 95 and row['Params (M)'] < 5:
        print("  ✓ Mobile deployment (high accuracy + lightweight)")
    elif row['Test Acc (%)'] >= 95:
        print("  ✓ Server-side deployment (high accuracy)")
    elif row['Time (min)'] < 300:
        print("  ✓ Quick prototyping (fast training)")
    elif row['Params (M)'] < 5:
        print("  ✓ Edge devices (small size)")
    else:
        print("  ⚠️ Not recommended for this task")

# Overall Ranking
print("\n" + "="*80)
print("FINAL RANKINGS")
print("="*80)

print("\n🏆 BY TEST ACCURACY:")
sorted_acc = df.sort_values('Test Acc (%)', ascending=False)
for i, (idx, row) in enumerate(sorted_acc.iterrows(), 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"  {medal} {row['Model']}: {row['Test Acc (%)']:.2f}%")

print("\n⚡ BY TRAINING SPEED:")
sorted_time = df.sort_values('Time (min)')
for i, (idx, row) in enumerate(sorted_time.iterrows(), 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"  {medal} {row['Model']}: {row['Time (min)']:.1f} minutes")

print("\n🎯 BY EFFICIENCY (Accuracy/Time):")
df['Efficiency'] = (df['Test Acc (%)'] / df['Time (min)']) * 10
sorted_eff = df.sort_values('Efficiency', ascending=False)
for i, (idx, row) in enumerate(sorted_eff.iterrows(), 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"  {medal} {row['Model']}: {row['Efficiency']:.3f}")

print("\n📦 BY MODEL SIZE (Smallest):")
sorted_size = df.sort_values('Params (M)')
for i, (idx, row) in enumerate(sorted_size.iterrows(), 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"  {medal} {row['Model']}: {row['Params (M)']:.1f}M parameters")

# Final Recommendation
print("\n" + "="*80)
print("🎯 RECOMMENDATION FOR YOUR PROJECT")
print("="*80)

best_model = sorted_acc.iloc[0]
print(f"\n✅ RECOMMENDED MODEL: {best_model['Model']}")
print(f"\nREASONS:")
print(f"  1. Highest test accuracy: {best_model['Test Acc (%)']:.2f}%")
print(f"  2. Good generalization (Train-Val gap: {best_model['Train Acc (%)'] - best_model['Val Acc (%)']:.2f}%)")
print(f"  3. Reasonable training time: {best_model['Time (min)']:.1f} minutes")
print(f"  4. Model size: {best_model['Params (M)']:.1f}M parameters")

print(f"\n📱 FOR DEPLOYMENT:")
print(f"  • Production/Research: Use {best_model['Model']}")
print(f"  • Mobile/Edge Devices: Consider MobileNetV2 or EfficientNetB0")
print(f"  • Quick Iteration: Use the fastest training model")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)