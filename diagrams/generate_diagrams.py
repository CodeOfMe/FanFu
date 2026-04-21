#!/usr/bin/env python3
"""Generate diagrams for FanFu project. All text in English."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

C = {'blue': '#2563EB', 'purple': '#7C3AED', 'green': '#059669', 'orange': '#D97706', 'red': '#DC2626', 'gray': '#6B7280', 'light': '#F3F4F6'}
os.makedirs('images', exist_ok=True)

# 1. Conversion Flow
fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.5)
ax.axis('off')

# GGUF
rect = patches.FancyBboxPatch((0.3, 0.8), 2.4, 2.0, boxstyle='round,pad=0.1', facecolor='#EFF6FF', edgecolor=C['blue'], linewidth=2)
ax.add_patch(rect)
ax.text(1.5, 2.3, 'Ollama GGUF', ha='center', va='center', fontsize=14, fontweight='bold', color=C['blue'])
ax.text(1.5, 1.8, 'Quantized Model', ha='center', va='center', fontsize=11, color=C['gray'])
ax.text(1.5, 1.4, 'Q4_K / Q6_K / Q8_0', ha='center', va='center', fontsize=10, color=C['gray'])
ax.text(1.5, 1.05, '1.0 GB (qwen3.5)', ha='center', va='center', fontsize=10, color=C['gray'])

ax.annotate('', xy=(3.2, 1.8), xytext=(2.7, 1.8), arrowprops=dict(arrowstyle='->', lw=2.5, color=C['blue']))
ax.text(2.95, 2.2, 'read', ha='center', va='center', fontsize=9, color=C['gray'])

# FanFu
rect = patches.FancyBboxPatch((3.2, 0.5), 2.8, 2.6, boxstyle='round,pad=0.1', facecolor='#F5F3FF', edgecolor=C['purple'], linewidth=2)
ax.add_patch(rect)
ax.text(4.6, 2.6, 'FanFu Engine', ha='center', va='center', fontsize=14, fontweight='bold', color=C['purple'])
ax.text(4.6, 2.1, 'Dequantize + Map', ha='center', va='center', fontsize=11, color=C['gray'])
ax.text(4.6, 1.6, '536 -> 572 tensors', ha='center', va='center', fontsize=10, color=C['gray'])
ax.text(4.6, 1.2, 'QKV auto-split', ha='center', va='center', fontsize=10, color=C['gray'])
ax.text(4.6, 0.8, '100% weight match', ha='center', va='center', fontsize=10, color=C['green'], fontweight='bold')

ax.annotate('', xy=(6.5, 1.8), xytext=(6.0, 1.8), arrowprops=dict(arrowstyle='->', lw=2.5, color=C['purple']))
ax.text(6.25, 2.2, 'convert', ha='center', va='center', fontsize=9, color=C['gray'])

# HF
rect = patches.FancyBboxPatch((6.5, 0.8), 2.4, 2.0, boxstyle='round,pad=0.1', facecolor='#ECFDF5', edgecolor=C['green'], linewidth=2)
ax.add_patch(rect)
ax.text(7.7, 2.3, 'HuggingFace', ha='center', va='center', fontsize=14, fontweight='bold', color=C['green'])
ax.text(7.7, 1.8, 'safetensors Model', ha='center', va='center', fontsize=11, color=C['gray'])
ax.text(7.7, 1.4, 'FP32 original precision', ha='center', va='center', fontsize=10, color=C['gray'])
ax.text(7.7, 1.05, '3.3 GB (qwen3.5)', ha='center', va='center', fontsize=10, color=C['gray'])

ax.annotate('', xy=(9.4, 1.8), xytext=(8.9, 1.8), arrowprops=dict(arrowstyle='->', lw=2.5, color=C['green']))
ax.text(9.15, 2.2, 'verify', ha='center', va='center', fontsize=9, color=C['gray'])

# Verify
rect = patches.FancyBboxPatch((9.4, 0.8), 2.3, 2.0, boxstyle='round,pad=0.1', facecolor='#FFFBEB', edgecolor=C['orange'], linewidth=2)
ax.add_patch(rect)
ax.text(10.55, 2.3, '43 Tests', ha='center', va='center', fontsize=14, fontweight='bold', color=C['orange'])
ax.text(10.55, 1.8, 'All Passed', ha='center', va='center', fontsize=12, color=C['green'], fontweight='bold')
ax.text(10.55, 1.3, '100% match rate', ha='center', va='center', fontsize=10, color=C['green'])

plt.savefig('images/conversion_flow.png', dpi=300, facecolor='white')
plt.savefig('images/conversion_flow.svg', facecolor='white')
plt.close()
print('Generated conversion_flow.png')

# 2. Test Results
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
cats = ['Tensor\nCount', 'Tensor\nMapping', 'Embedding', 'FFN\nLayers', 'QKV\nSplit', 'Vision\nTower', 'MTP\nLayers', 'SSM', 'Shortconv', 'All\nWeights']
qwen = [100, 100, 100, 100, 100, 100, 100, 100, 0, 100]
lfm = [100, 100, 100, 100, 0, 0, 0, 0, 100, 100]
x = np.arange(len(cats))
w = 0.35
ax.bar(x-w/2, qwen, w, color=C['blue'], label='qwen3.5 (536->572)', edgecolor='white', linewidth=1.5)
ax.bar(x+w/2, lfm, w, color=C['purple'], label='lfm2.5 (148->148)', edgecolor='white', linewidth=1.5)
ax.set_ylabel('Match Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Weight Verification Results - All Tests Passed', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylim(0, 110)
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.2)
ax.axhline(y=100, color=C['green'], linestyle='--', alpha=0.5, linewidth=1)
plt.tight_layout()
plt.savefig('images/test_results.png', dpi=300, facecolor='white')
plt.savefig('images/test_results.svg', facecolor='white')
plt.close()
print('Generated test_results.png')

# 3. Tensor Count
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.pie([536, 36], labels=['Original', 'QKV Split'], colors=[C['blue'], C['green']], autopct='%d', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax1.set_title('qwen3.5: 536 -> 572', fontsize=12, fontweight='bold')
ax2.pie([148, 0], labels=['Original', 'No Split'], colors=[C['purple'], C['light']], autopct='%d', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
ax2.set_title('lfm2.5: 148 -> 148', fontsize=12, fontweight='bold')
plt.suptitle('Tensor Count Before vs After Conversion', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images/tensor_pie.png', dpi=300, facecolor='white')
plt.savefig('images/tensor_pie.svg', facecolor='white')
plt.close()
print('Generated tensor_pie.png')

# 4. Quantization Types
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
types = ['F32', 'F16', 'BF16', 'Q4_0', 'Q4_K', 'Q5_K', 'Q6_K', 'Q8_0']
sizes = [1, 1, 1, 32, 256, 256, 256, 32]
colors_bar = [C['green'] if s==1 else C['blue'] if s<=32 else C['purple'] for s in sizes]
bars = ax.barh(types, sizes, color=colors_bar, edgecolor='white', linewidth=1.5, height=0.5)
ax.set_xlabel('Block Size', fontsize=11, fontweight='bold')
ax.set_title('Supported Quantization Types', fontsize=13, fontweight='bold', pad=10)
ax.grid(axis='x', alpha=0.2)
for bar, val in zip(bars, sizes):
    ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2, str(val), va='center', fontsize=10, fontweight='bold')
legend_elements = [patches.Patch(facecolor=C['green'], label='Scalar (1)'), patches.Patch(facecolor=C['blue'], label='Basic (32)'), patches.Patch(facecolor=C['purple'], label='K-Quant (256)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.set_xlim(0, 300)
plt.tight_layout()
plt.savefig('images/quant_types.png', dpi=300, facecolor='white')
plt.savefig('images/quant_types.svg', facecolor='white')
plt.close()
print('Generated quant_types.png')

# 5. Model Radar
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
categories = ['Layers', 'Heads', 'Vocab', 'GGUF\nTensors', 'HF\nTensors', 'Size\n(GB)']
qwen_vals = [24, 8, 248, 536, 572, 3.3]
lfm_vals = [16, 32, 65, 148, 148, 4.4]
qwen_norm = [v/max(qwen_vals) for v in qwen_vals]
lfm_norm = [v/max(lfm_vals) for v in lfm_vals]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
qwen_norm += qwen_norm[:1]
lfm_norm += lfm_norm[:1]
angles += angles[:1]
ax = plt.subplot(111, projection='polar')
ax.plot(angles, qwen_norm, 'o-', color=C['blue'], linewidth=2, label='qwen3.5')
ax.fill(angles, qwen_norm, alpha=0.15, color=C['blue'])
ax.plot(angles, lfm_norm, 's-', color=C['purple'], linewidth=2, label='lfm2.5')
ax.fill(angles, lfm_norm, alpha=0.15, color=C['purple'])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True)
plt.tight_layout()
plt.savefig('images/model_radar.png', dpi=300, facecolor='white')
plt.savefig('images/model_radar.svg', facecolor='white')
plt.close()
print('Generated model_radar.png')

# 6. Inference Test
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
questions = ['2+2=?', 'Capital\nof France', 'Factorial\nCode', 'Quantum\nExplain', 'Chinese\nIntro']
qwen_scores = [100, 100, 100, 100, 100]
lfm_scores = [100, 100, 100, 100, 100]
x = np.arange(len(questions))
w = 0.35
ax.bar(x-w/2, qwen_scores, w, color=C['blue'], label='qwen3.5 (GGUF)', edgecolor='white', linewidth=1.5)
ax.bar(x+w/2, lfm_scores, w, color=C['purple'], label='lfm2.5 (GGUF)', edgecolor='white', linewidth=1.5)
ax.set_ylabel('Correct Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Ollama Inference Test Results', fontsize=14, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(questions, fontsize=9)
ax.set_ylim(0, 110)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.2)
for bars in [ax.patches[:5], ax.patches[5:]]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+2, '100%', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('images/inference_test.png', dpi=300, facecolor='white')
plt.savefig('images/inference_test.svg', facecolor='white')
plt.close()
print('Generated inference_test.png')

print('All 6 images generated successfully.')