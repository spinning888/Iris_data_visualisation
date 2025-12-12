import numpy as np
import json
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from skimage import measure
import os

# ==========================================================
# Global Settings
# ==========================================================
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ['#2E86AB', '#A23B72', '#F18F01']  # Deep blue, purple, orange
COLORS_COOL = ['#1E90FF', '#00CED1', '#32CD32']  # Cool colors
CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']


def task4_html_interactive():
    """生成Task 4的交互式HTML可视化"""
    
    print("\n" + "="*60)
    print("TASK 4: 3D Interactive HTML Visualization")
    print("="*60)
    
    iris = load_iris()
    X_full = iris.data
    y = iris.target
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_3d_scaled = X_scaled[:, [1, 2, 3]]  # Sepal Width, Petal Length, Petal Width
    
    # === 4-A: Decision Boundary Surface ===
    print("\n  [4-A] Building 3D Decision Boundary Surface...")
    
    clf = LogisticRegression(max_iter=500, multi_class='auto')
    clf.fit(X_3d_scaled, y)
    train_acc = clf.score(X_3d_scaled, y)
    
    # Create fig for 4A
    fig_4a = go.Figure()
    
    # Create grid
    resolution = 20
    mins = X_3d_scaled.min(axis=0) - 0.5
    maxs = X_3d_scaled.max(axis=0) + 0.5
    
    xx = np.linspace(mins[0], maxs[0], resolution)
    yy = np.linspace(mins[1], maxs[1], resolution)
    zz = np.linspace(mins[2], maxs[2], resolution)
    
    XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing='ij')
    grid_points = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]
    
    predictions = clf.predict(grid_points)
    vol = predictions.reshape(XX.shape).astype(float)
    vol_smooth = gaussian_filter(vol, sigma=1.0)
    
    # Extract surfaces using marching cubes
    for class_idx in range(3):
        threshold = class_idx + 0.5
        try:
            verts, faces, _, _ = measure.marching_cubes(vol_smooth, level=threshold)
            
            if len(verts) > 0:
                # Map verts back to original space
                verts_mapped = np.zeros_like(verts)
                for dim in range(3):
                    verts_mapped[:, dim] = mins[dim] + verts[:, dim] * (maxs[dim] - mins[dim]) / (resolution - 1)
                
                # Create triangulation
                fig_4a.add_trace(go.Scatter3d(
                    x=verts_mapped[faces][:, 0].flatten(),
                    y=verts_mapped[faces][:, 1].flatten(),
                    z=verts_mapped[faces][:, 2].flatten(),
                    mode='markers',
                    marker=dict(size=2, color=COLORS[class_idx], opacity=0.3),
                    name=CLASS_NAMES[class_idx],
                    hoverinfo='skip'
                ))
        except Exception as e:
            print(f"    Warning: Could not extract surface for class {class_idx}")
    
    # Add data points
    for class_idx in range(3):
        pts = X_3d_scaled[y == class_idx]
        fig_4a.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=5, color=COLORS_COOL[class_idx]),
            name=f'{CLASS_NAMES[class_idx]}',
            hovertemplate='<b>%{fullData.name}</b><br>SepalWidth: %{x:.2f}<br>PetalLength: %{y:.2f}<br>PetalWidth: %{z:.2f}<extra></extra>'
        ))
    
    fig_4a.update_layout(
        title=f'<b>Task 4-A: 3D Decision Boundary Surface</b><br><sub>Accuracy: {train_acc*100:.1f}%</sub>',
        scene=dict(
            xaxis_title='Sepal Width',
            yaxis_title='Petal Length',
            zaxis_title='Petal Width',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        width=900,
        height=800,
        font=dict(family='Times New Roman', size=12),
        hovermode='closest'
    )
    
    # === 4-B: Probability Contour Map ===
    print("  [4-B] Building 3D Probability Contour Map...")
    
    svm_multi = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
    svm_multi.fit(X_3d_scaled, y)
    
    # Create fine mesh
    RES = 80
    pad = 1.5
    x_min, x_max = X_3d_scaled[:, 0].min() - pad, X_3d_scaled[:, 0].max() + pad
    y_min, y_max = X_3d_scaled[:, 1].min() - pad, X_3d_scaled[:, 1].max() + pad
    
    xx_fine = np.linspace(x_min, x_max, RES)
    yy_fine = np.linspace(y_min, y_max, RES)
    XX_fine, YY_fine = np.meshgrid(xx_fine, yy_fine)
    
    z_fixed = X_3d_scaled[:, 2].mean()
    grid_fine = np.c_[XX_fine.ravel(), YY_fine.ravel(), np.full(XX_fine.size, z_fixed)]
    probs = svm_multi.predict_proba(grid_fine)
    Z_prob = probs[:, 1].reshape(XX_fine.shape) * 100
    
    fig_4b = go.Figure()
    
    # Probability surface
    fig_4b.add_trace(go.Surface(
        x=XX_fine,
        y=YY_fine,
        z=Z_prob,
        colorscale='RdYlBu_r',
        name='Probability Surface',
        hovertemplate='SepalWidth: %{x:.2f}<br>PetalLength: %{y:.2f}<br>Probability: %{z:.1f}%<extra></extra>'
    ))
    
    # Add data points at z=0
    for class_idx in range(3):
        pts = X_3d_scaled[y == class_idx]
        fig_4b.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=np.zeros(len(pts)),
            mode='markers',
            marker=dict(size=5, color=COLORS_COOL[class_idx]),
            name=f'{CLASS_NAMES[class_idx]}',
            hovertemplate='<b>%{fullData.name}</b><br>SepalWidth: %{x:.2f}<br>PetalLength: %{y:.2f}<extra></extra>'
        ))
    
    fig_4b.update_layout(
        title='<b>Task 4-B: 3D Probability Contour Map</b><br><sub>Versicolor Probability</sub>',
        scene=dict(
            xaxis_title='Sepal Width',
            yaxis_title='Petal Length',
            zaxis_title='Probability (%)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            zaxis=dict(range=[0, 100])
        ),
        width=900,
        height=800,
        font=dict(family='Times New Roman', size=12),
        hovermode='closest'
    )
    
    # === Create combined HTML ===
    print("\n  Creating HTML file...")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Task 4: Iris Classification - 3D Interactive Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Times New Roman', Times, serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{
                text-align: center;
                color: #2E86AB;
                margin-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            .section {{
                margin-bottom: 40px;
                page-break-inside: avoid;
            }}
            .plot-container {{
                border: 1px solid #ddd;
                border-radius: 6px;
                margin-top: 15px;
                background-color: #fafafa;
            }}
            .description {{
                background-color: #f0f7ff;
                border-left: 4px solid #2E86AB;
                padding: 12px 15px;
                margin-bottom: 15px;
                border-radius: 4px;
                font-size: 13px;
                line-height: 1.6;
            }}
            .legend {{
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                margin-top: 15px;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 4px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 3px;
                border: 1px solid #999;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #999;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Task 4: Iris Classification - 3D Interactive Visualization</h1>
            <p class="subtitle">Iris Dataset | 3D Decision Boundary & Probability Contour Maps</p>
            
            <!-- Section 4-A -->
            <div class="section">
                <h2>4-A: 3D Decision Boundary Surface (Marching Cubes)</h2>
                <div class="description">
                    <b>说明：</b> 使用 Logistic Regression 多分类模型，通过 Marching Cubes 算法提取三个分类的决策边界曲面。
                    每个颜色表示一个分类的边界面，数据点投影在3D空间中显示。
                    <br><b>交互：</b> 鼠标拖动旋转 | 滚轮缩放 | 点击图例显/隐
                </div>
                <div id="plot4a" class="plot-container"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #1E90FF;"></div>
                        <span>Setosa</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #00CED1;"></div>
                        <span>Versicolor</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #32CD32;"></div>
                        <span>Virginica</span>
                    </div>
                </div>
            </div>
            
            <!-- Section 4-B -->
            <div class="section">
                <h2>4-B: 3D Probability Contour Map</h2>
                <div class="description">
                    <b>说明：</b> 使用 RBF SVM 训练概率模型，绘制 Versicolor 分类的概率等高面。
                    红色表示高概率区域，蓝色表示低概率区域。底层的数据点显示训练样本的分布。
                    <br><b>交互：</b> 鼠标拖动旋转 | 滚轮缩放 | 点击图例显/隐
                </div>
                <div id="plot4b" class="plot-container"></div>
            </div>
            
            <div class="footer">
                <p>✓ Generated by Project3_Final.py | Iris Dataset Classification Visualization</p>
            </div>
        </div>
        
        <script>
            // 4-A Plot
            const plot4aData = {json.dumps(fig_4a.to_json())};
            Plotly.newPlot('plot4a', plot4aData.data, plot4aData.layout, {{responsive: true}});
            
            // 4-B Plot
            const plot4bData = {json.dumps(fig_4b.to_json())};
            Plotly.newPlot('plot4b', plot4bData.data, plot4bData.layout, {{responsive: true}});
        </script>
    </body>
    </html>
    """
    
    # Save HTML
    html_path = os.path.join(OUTPUT_DIR, 'Task4_Interactive.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[OK] Saved: Task4_Interactive.html")
    print(f"     Location: {os.path.abspath(html_path)}")


if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("Task 4: HTML Interactive Visualization")
    print("="*60)
    
    iris = load_iris()
    X_full = iris.data
    y = iris.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    X_3d_scaled = X_scaled[:, [1, 2, 3]]
    
    # === 4-A: Decision Boundary ===
    print("\n  [4-A] Building 3D Decision Boundary Surface...")
    
    clf = LogisticRegression(max_iter=500, multi_class='auto')
    clf.fit(X_3d_scaled, y)
    train_acc = clf.score(X_3d_scaled, y)
    
    fig_4a = go.Figure()
    
    resolution = 20
    mins = X_3d_scaled.min(axis=0) - 0.5
    maxs = X_3d_scaled.max(axis=0) + 0.5
    
    xx = np.linspace(mins[0], maxs[0], resolution)
    yy = np.linspace(mins[1], maxs[1], resolution)
    zz = np.linspace(mins[2], maxs[2], resolution)
    
    XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing='ij')
    grid_points = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]
    
    predictions = clf.predict(grid_points)
    vol = predictions.reshape(XX.shape).astype(float)
    vol_smooth = gaussian_filter(vol, sigma=1.0)
    
    for class_idx in range(3):
        threshold = class_idx + 0.5
        try:
            verts, faces, _, _ = measure.marching_cubes(vol_smooth, level=threshold)
            
            if len(verts) > 0:
                verts_mapped = np.zeros_like(verts)
                for dim in range(3):
                    verts_mapped[:, dim] = mins[dim] + verts[:, dim] * (maxs[dim] - mins[dim]) / (resolution - 1)
                
                x_vals = verts_mapped[:, 0]
                y_vals = verts_mapped[:, 1]
                z_vals = verts_mapped[:, 2]
                
                fig_4a.add_trace(go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers',
                    marker=dict(size=1.5, color=COLORS[class_idx], opacity=0.4),
                    name=CLASS_NAMES[class_idx],
                    hoverinfo='skip'
                ))
        except:
            print(f"    Warning: Could not extract surface for class {class_idx}")
    
    for class_idx in range(3):
        pts = X_3d_scaled[y == class_idx]
        fig_4a.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode='markers',
            marker=dict(size=6, color=COLORS_COOL[class_idx], symbol='circle'),
            name=f'{CLASS_NAMES[class_idx]}',
            text=[f"{CLASS_NAMES[class_idx]}<br>X:{x:.2f}<br>Y:{y:.2f}<br>Z:{z:.2f}" 
                  for x, y, z in zip(pts[:, 0], pts[:, 1], pts[:, 2])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig_4a.update_layout(
        title=f'<b>Task 4-A: 3D Decision Boundary Surface</b><br><sub>Accuracy: {train_acc*100:.1f}%</sub>',
        scene=dict(
            xaxis_title='Sepal Width',
            yaxis_title='Petal Length',
            zaxis_title='Petal Width',
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.2))
        ),
        width=900,
        height=800,
        font=dict(family='Times New Roman', size=12),
        hovermode='closest',
        showlegend=True
    )
    
    # === 4-B: Probability Map ===
    print("  [4-B] Building 3D Probability Contour Map...")
    
    svm_multi = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
    svm_multi.fit(X_3d_scaled, y)
    
    RES = 80
    pad = 1.5
    x_min, x_max = X_3d_scaled[:, 0].min() - pad, X_3d_scaled[:, 0].max() + pad
    y_min, y_max = X_3d_scaled[:, 1].min() - pad, X_3d_scaled[:, 1].max() + pad
    
    xx_fine = np.linspace(x_min, x_max, RES)
    yy_fine = np.linspace(y_min, y_max, RES)
    XX_fine, YY_fine = np.meshgrid(xx_fine, yy_fine)
    
    z_fixed = X_3d_scaled[:, 2].mean()
    grid_fine = np.c_[XX_fine.ravel(), YY_fine.ravel(), np.full(XX_fine.size, z_fixed)]
    probs = svm_multi.predict_proba(grid_fine)
    Z_prob = probs[:, 1].reshape(XX_fine.shape) * 100
    
    fig_4b = go.Figure()
    
    fig_4b.add_trace(go.Surface(
        x=XX_fine,
        y=YY_fine,
        z=Z_prob,
        colorscale='RdYlBu_r',
        name='Probability',
        colorbar=dict(title='Prob (%)', thickness=15, len=0.7),
        hovertemplate='Sepal Width: %{x:.2f}<br>Petal Length: %{y:.2f}<br>Probability: %{z:.1f}%<extra></extra>'
    ))
    
    for class_idx in range(3):
        pts = X_3d_scaled[y == class_idx]
        fig_4b.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=np.zeros(len(pts)),
            mode='markers',
            marker=dict(size=6, color=COLORS_COOL[class_idx], symbol='circle'),
            name=f'{CLASS_NAMES[class_idx]}',
            text=[f"{CLASS_NAMES[class_idx]}<br>X:{x:.2f}<br>Y:{y:.2f}" 
                  for x, y in zip(pts[:, 0], pts[:, 1])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig_4b.update_layout(
        title='<b>Task 4-B: 3D Probability Contour Map</b><br><sub>Versicolor Class</sub>',
        scene=dict(
            xaxis_title='Sepal Width',
            yaxis_title='Petal Length',
            zaxis_title='Probability (%)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            zaxis=dict(range=[0, 100])
        ),
        width=900,
        height=800,
        font=dict(family='Times New Roman', size=12),
        hovermode='closest',
        showlegend=True
    )
    
    # === Generate HTML ===
    print("\n  Creating HTML file...")
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task 4: Iris Classification - 3D Interactive Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Times New Roman', Times, serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            padding: 40px;
        }}
        h1 {{
            text-align: center;
            color: #2E86AB;
            margin-bottom: 8px;
            font-size: 28px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 14px;
        }}
        .section {{
            margin-bottom: 50px;
            page-break-inside: avoid;
        }}
        h2 {{
            color: #2E86AB;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .plot-container {{
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-top: 15px;
            background-color: #fafafa;
            overflow: hidden;
        }}
        .description {{
            background-color: #f0f7ff;
            border-left: 4px solid #2E86AB;
            padding: 12px 15px;
            margin-bottom: 15px;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.6;
            color: #333;
        }}
        .legend {{
            display: flex;
            gap: 25px;
            flex-wrap: wrap;
            margin-top: 15px;
            padding: 12px;
            background-color: #f9f9f9;
            border-radius: 4px;
            border: 1px solid #eee;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #999;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #999;
            font-size: 12px;
        }}
        .info-box {{
            background-color: #fff9e6;
            border-left: 4px solid #F18F01;
            padding: 12px 15px;
            margin-top: 10px;
            border-radius: 4px;
            font-size: 12px;
            color: #555;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Task 4: Iris Classification - 3D Interactive Visualization</h1>
        <p class="subtitle">Iris Dataset | Decision Boundary & Probability Contour Maps</p>
        
        <!-- Section 4-A -->
        <div class="section">
            <h2>📍 4-A: 3D Decision Boundary Surface (Marching Cubes)</h2>
            <div class="description">
                <b>Model:</b> Logistic Regression (Multi-class) | <b>Algorithm:</b> Marching Cubes Surface Extraction
                <br><b>说明：</b> 使用 Logistic Regression 多分类模型，通过 Marching Cubes 算法提取三个分类的决策边界曲面。
                每个颜色表示一个分类的边界面，数据点显示训练样本的分布。
                <br><b>交互：</b> 鼠标拖动旋转 | 滚轮缩放 | 点击图例显/隐
            </div>
            <div id="plot4a" class="plot-container"></div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #2E86AB;"></div>
                    <span><b>Setosa</b></span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #A23B72;"></div>
                    <span><b>Versicolor</b></span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #F18F01;"></div>
                    <span><b>Virginica</b></span>
                </div>
            </div>
            <div class="info-box">
                ℹ️ 准确率: <b>{train_acc*100:.1f}%</b> | 特征: Sepal Width, Petal Length, Petal Width
            </div>
        </div>
        
        <!-- Section 4-B -->
        <div class="section">
            <h2>📈 4-B: 3D Probability Contour Map</h2>
            <div class="description">
                <b>Model:</b> SVM with RBF Kernel (Probability Calibrated) | <b>Target:</b> Versicolor Classification
                <br><b>说明：</b> 使用 RBF SVM 训练概率模型，绘制 Versicolor 分类的概率等高面。
                红色表示高概率区域，蓝色表示低概率区域。底层的数据点显示训练样本的分布。
                <br><b>交互：</b> 鼠标拖动旋转 | 滚轮缩放 | 点击图例显/隐
            </div>
            <div id="plot4b" class="plot-container"></div>
        </div>
        
        <div class="footer">
            <p>✓ Generated by task4_html.py | Iris Dataset Classification Visualization | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
    
    <script>
        // 4-A Plot
        const plot4aData = {json.dumps(fig_4a.to_json())};
        Plotly.newPlot('plot4a', plot4aData.data, plot4aData.layout, {{responsive: true}});
        
        // 4-B Plot
        const plot4bData = {json.dumps(fig_4b.to_json())};
        Plotly.newPlot('plot4b', plot4bData.data, plot4bData.layout, {{responsive: true}});
    </script>
</body>
</html>
"""
    
    html_path = os.path.join(OUTPUT_DIR, 'Task4_Interactive.html')
    
    import datetime
    html_content = html_content.replace(
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n[OK] Saved: Task4_Interactive.html")
    print(f"     Location: {os.path.abspath(html_path)}")
    print("\n" + "="*60)
    print("✓ HTML Interactive Visualization Complete!")
    print("="*60 + "\n")
