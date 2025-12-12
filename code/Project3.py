import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from skimage import measure
from scipy.ndimage import gaussian_filter
import os

# ==========================================================
# Global Settings
# ==========================================================
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Professional color scheme (high contrast, soft)
COLORS = ['#2E86AB', '#A23B72', '#F18F01']  # Deep blue, purple, orange
COLORS_COOL = ['#1E90FF', '#00CED1', '#32CD32']  # 冷色调：Dodger Blue, Dark Turquoise, Lime Green
CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']
MARKERS = ['o', 's', '^']

# Typography standards
FONT_TITLE_MAIN = 16
FONT_TITLE_SUB = 13
FONT_LABEL = 11
FONT_LEGEND = 10

# Output directory
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(filename, fig=None, dpi=300, save_pdf=True):
    """Save figure as PNG (high quality) and optionally PDF (vector)"""
    if fig is None:
        fig = plt.gcf()
    
    base_name = filename.replace('.png', '').replace('.pdf', '')
    
    # PNG (high resolution)
    png_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
    try:
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', format='png')
        print(f"[OK] Saved: {base_name}.png", end="")
    except Exception as e:
        print(f"[FAIL] PNG save failed: {e}")
        return
    
    # PDF (vector format for papers) - optional
    if save_pdf:
        pdf_path = os.path.join(OUTPUT_DIR, f"{base_name}.pdf")
        try:
            fig.savefig(pdf_path, dpi=dpi, bbox_inches='tight', format='pdf')
            print(" & .pdf")
        except Exception as e:
            print(f" (PDF save skipped: {type(e).__name__})")
    else:
        print()


# ==========================================================
# TASK 1: 2D Decision Boundary (3 classifiers × 4 subplots)
# ==========================================================
def task1_2d_decision_boundary():
    print("\n" + "="*60)
    print("TASK 1: 2D Decision Boundary Comparison")
    print("="*60)
    
    iris = load_iris()
    X = iris.data[:, [2, 3]]  # Petal Length, Petal Width
    y = iris.target
    feature_names = ['Petal Length (cm)', 'Petal Width (cm)']
    
    classifiers = [
        (LogisticRegression(solver='lbfgs', max_iter=200), 
         "Logistic Regression"),
        (SVC(kernel='rbf', probability=True, gamma=1, C=1), 
         "RBF SVM"),
        (DecisionTreeClassifier(max_depth=4, random_state=42), 
         "Decision Tree")
    ]
    
    # 高分辨率网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # 使用 GridSpec 创建复杂布局
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08, left=0.05, right=0.95)
    
    fig.suptitle('Task 1: Comparison of Classifiers on Iris (3-class 2D)',
                fontsize=14, fontweight='bold', y=0.98)
    
    colors_scatter = ['#1f77b4', '#d62728', '#2ca02c']
    colors_decision = ['#87CEEB', '#FFB6C6', '#90EE90']
    cmap_scatter = ListedColormap(colors_scatter)
    
    for row, (clf, clf_name) in enumerate(classifiers):
        print(f"  Training {clf_name}...")
        
        clf.fit(X, y)
        train_acc = clf.score(X, y)
        
        # 预测
        Z = clf.predict(grid).reshape(xx.shape)
        probs = clf.predict_proba(grid).reshape(xx.shape[0], xx.shape[1], 3)
        
        # 决策边界
        ax_boundary = fig.add_subplot(gs[row, 0])
        cmap_decision = ListedColormap(colors_decision)
        ax_boundary.contourf(xx, yy, Z, levels=3, cmap=cmap_decision, alpha=0.4)
        ax_boundary.contour(xx, yy, Z, levels=2, colors='black', linewidths=0.5, alpha=0.5)
        scatter = ax_boundary.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scatter, 
                                      edgecolors='black', s=50, linewidth=0.7, alpha=0.9)
        ax_boundary.set_xlim(xx.min(), xx.max())
        ax_boundary.set_ylim(yy.min(), yy.max())
        ax_boundary.set_title(f'{clf_name}\nAccuracy: {train_acc*100:.1f}%', 
                             fontsize=11, fontweight='bold', color='#2E86AB')
        ax_boundary.set_xlabel(feature_names[0], fontsize=9)
        ax_boundary.set_ylabel(feature_names[1], fontsize=9)
        ax_boundary.grid(True, alpha=0.2, linestyle='--')
        ax_boundary.tick_params(labelsize=8)
        
        if row == 0:
            ax_boundary.legend(handles=scatter.legend_elements()[0], 
                             labels=CLASS_NAMES, loc='upper right', fontsize=8)
        
        # 三个概率图
        for c in range(3):
            ax_prob = fig.add_subplot(gs[row, c + 1])
            
            cmap_prob = LinearSegmentedColormap.from_list(
                f'prob_{c}', ['white', colors_decision[c]], N=256
            )
            contour = ax_prob.contourf(xx, yy, probs[:, :, c], 
                                      levels=12, cmap=cmap_prob, alpha=0.85)
            ax_prob.contour(xx, yy, probs[:, :, c], levels=[0.5], colors='black', 
                           linewidths=1, alpha=0.4)  # 50% 等概率线
            ax_prob.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_scatter, 
                           edgecolors='black', s=30, linewidth=0.5, alpha=0.8)
            ax_prob.set_xlim(xx.min(), xx.max())
            ax_prob.set_ylim(yy.min(), yy.max())
            ax_prob.set_title(f'P({CLASS_NAMES[c]})', fontsize=10, fontweight='bold')
            ax_prob.set_xlabel(feature_names[0], fontsize=9)
            ax_prob.set_ylabel(feature_names[1], fontsize=9)
            cbar = fig.colorbar(contour, ax=ax_prob, shrink=0.8, ticks=[0, 0.5, 1])
            cbar.ax.tick_params(labelsize=7)
            ax_prob.grid(True, alpha=0.2, linestyle='--')
            ax_prob.tick_params(labelsize=8)
    
    save_figure('Task1_Classifier_Comparison', fig)
    plt.close(fig)


# ==========================================================
# TASK 2: 3D Binary Decision Surfaces (3 pairs, 1 figure)
# ==========================================================
def task2_3d_binary_surfaces():
    print("\n" + "="*60)
    print("TASK 2: 3D Binary Decision Surfaces (3 Class Pairs)")
    print("="*60)
    
    iris = load_iris()
    X_3d = iris.data[:, [1, 2, 3]]  # Sepal Width, Petal Length, Petal Width
    y = iris.target
    
    binary_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = [
        ('Setosa', 'Versicolor'),
        ('Setosa', 'Virginica'),
        ('Versicolor', 'Virginica')
    ]
    
    fig = plt.figure(figsize=(18, 5.5))
    
    for pair_idx, ((c1, c2), (name1, name2)) in enumerate(zip(binary_pairs, pair_names)):
        print(f"  → Processing pair {pair_idx+1}: {name1} vs {name2}")
        
        # Select binary class
        mask = (y == c1) | (y == c2)
        X_binary = X_3d[mask]
        y_binary = (y[mask] == c2).astype(int)
        
        # Train logistic regression
        model = LogisticRegression(max_iter=300)
        model.fit(X_binary, y_binary)
        
        train_acc = model.score(X_binary, y_binary)
        
        # Create 3D mesh
        x = np.linspace(X_binary[:, 0].min() - 0.5, X_binary[:, 0].max() + 0.5, 25)
        y_mesh = np.linspace(X_binary[:, 1].min() - 0.5, X_binary[:, 1].max() + 0.5, 25)
        Xg, Yg = np.meshgrid(x, y_mesh)
        
        w, b = model.coef_[0], model.intercept_[0]
        if abs(w[2]) > 1e-6:
            Zg = -(w[0] * Xg + w[1] * Yg + b) / w[2]
        else:
            Zg = np.full_like(Xg, X_binary[:, 2].mean())
        
        # Add subplot
        ax = fig.add_subplot(1, 3, pair_idx + 1, projection='3d')
        
        # Decision surface with cool color gradient
        ax.plot_surface(Xg, Yg, Zg, alpha=0.35, color='#E0F2FF', edgecolor='none')
        # Add wireframe for better visualization
        ax.plot_wireframe(Xg, Yg, Zg, rstride=3, cstride=3, 
                         color='#87CEEB', linewidth=0.8, alpha=0.4)
        
        # Data points with cool colors
        ax.scatter(X_binary[y_binary == 0, 0], X_binary[y_binary == 0, 1], X_binary[y_binary == 0, 2],
                  c=COLORS_COOL[0], label=name1, s=80, edgecolor='black', linewidth=1.2, alpha=0.85)
        ax.scatter(X_binary[y_binary == 1, 0], X_binary[y_binary == 1, 1], X_binary[y_binary == 1, 2],
                  c=COLORS_COOL[1], label=name2, s=80, edgecolor='black', linewidth=1.2, alpha=0.85)
        
        # Styling
        ax.set_title(f'{name1} vs {name2}\n{train_acc*100:.1f}%',
                    fontsize=FONT_TITLE_SUB, fontweight='bold')
        ax.set_xlabel('Sepal Width', fontsize=FONT_LABEL, fontweight='bold')
        ax.set_ylabel('Petal Length', fontsize=FONT_LABEL, fontweight='bold')
        ax.set_zlabel('Petal Width', fontsize=FONT_LABEL, fontweight='bold')
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
        ax.view_init(elev=20, azim=45)
        ax.tick_params(labelsize=9)
    
    fig.suptitle('Task 2: 3D Binary Decision Surfaces',
                fontsize=FONT_TITLE_MAIN, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.35, top=0.88)
    
    save_figure('Task2_Binary_Surfaces', fig)
    plt.close(fig)


# ==========================================================
# TASK 3: 3D Probability Contour Maps (3 pairs, 1 figure)
# ==========================================================
def task3_3d_probability_contours():
    print("\n" + "="*60)
    print("TASK 3: 3D Probability Contour Maps (3 Class Pairs)")
    print("="*60)
    
    iris = load_iris()
    X_3d = iris.data[:, [1, 2, 3]]  # Sepal Width, Petal Length, Petal Width
    y = iris.target
    
    binary_pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = [
        ('Setosa', 'Versicolor'),
        ('Setosa', 'Virginica'),
        ('Versicolor', 'Virginica')
    ]
    
    fig = plt.figure(figsize=(18, 5.5))
    
    for pair_idx, ((c1, c2), (name1, name2)) in enumerate(zip(binary_pairs, pair_names)):
        print(f"  → Processing pair {pair_idx+1}: {name1} vs {name2}")
        
        # Select binary class
        mask = (y == c1) | (y == c2)
        X_binary = X_3d[mask]
        y_binary = (y[mask] == c2).astype(int)
        
        # Train SVM for probability
        model = SVC(kernel='rbf', C=100, probability=True, gamma='scale')
        model.fit(X_binary, y_binary)
        
        # Create fine mesh
        RES = 120
        pad = 1.5
        x_min, x_max = X_binary[:, 0].min() - pad, X_binary[:, 0].max() + pad
        y_min, y_max = X_binary[:, 1].min() - pad, X_binary[:, 1].max() + pad
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, RES),
                             np.linspace(y_min, y_max, RES))
        
        # Fixed z-value (mean of third dimension)
        z_fixed = X_binary[:, 2].mean()
        
        # Predict probabilities
        grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z_fixed)]
        probs = model.predict_proba(grid)[:, 1].reshape(xx.shape) * 100
        
        # Add subplot
        ax = fig.add_subplot(1, 3, pair_idx + 1, projection='3d')
        
        # === High-contrast probability map styling ===
        z_base = np.zeros_like(xx)
        
        # 1. High-contrast baseline plane: DeepSkyBlue with opacity
        ax.plot_surface(xx, yy, z_base, color='deepskyblue', alpha=0.4, shade=False, zorder=1)
        
        # 2. White grid overlay on baseline
        ax.plot_wireframe(xx, yy, z_base, rstride=10, cstride=10, 
                         color='white', linewidth=0.5, alpha=0.6, zorder=2)
        
        # 3. Bold black decision line (50% probability threshold)
        ax.contour(xx, yy, probs, levels=[50], colors='black', linewidths=4, 
                  zdir='z', offset=0, zorder=10)
        
        # 4. Probability surface with RdYlBu colormap (对比度强)
        cmap_prob = plt.cm.RdYlBu_r  # 红黄蓝对比colormap（红=高概率，蓝=低概率）
        ax.plot_surface(xx, yy, probs, cmap=cmap_prob, alpha=0.75, 
                       shade=True, edgecolor='none', zorder=5)
        
        # 5. Dark skeleton wireframe
        ax.plot_wireframe(xx, yy, probs, rstride=6, cstride=6, 
                         color='midnightblue', linewidth=0.6, alpha=0.35, zorder=5)
        
        # 6. 四个侧面投影
        z_min = -100
        x_min = xx.min()
        x_max = xx.max()
        y_min = yy.min()
        cmap_prob = plt.cm.RdYlBu_r
        
        # 底面投影 (Z = -100)
        ax.contourf(xx, yy, probs, zdir='z', offset=z_min, cmap=cmap_prob, alpha=0.4, levels=12, zorder=0)
        
        # 左侧投影 (X = x_min)
        ax.contourf(xx, yy, probs, zdir='x', offset=x_min, cmap=cmap_prob, alpha=0.4, levels=12, zorder=0)
        
        # 正面投影 (Y = y_min)
        ax.contourf(xx, yy, probs, zdir='y', offset=y_min, cmap=cmap_prob, alpha=0.4, levels=12, zorder=0)
        
        # 右侧投影 (X = x_max)
        ax.contourf(xx, yy, probs, zdir='x', offset=x_max, cmap=cmap_prob, alpha=0.4, levels=12, zorder=0)
        
        # Project data points on base plane with cool colors
        for class_idx, class_c in enumerate([c1, c2]):
            pts = X_binary[y_binary == class_idx]
            ax.scatter(pts[:, 0], pts[:, 1], np.zeros(len(pts)),
                      c=COLORS_COOL[class_idx], s=60, edgecolor='black', linewidth=1.2,
                      alpha=0.85, zorder=10)
        
        # Styling
        ax.set_title(f'{name1} vs {name2}',
                    fontsize=FONT_TITLE_SUB, fontweight='bold')
        ax.set_xlabel('Sepal Width', fontsize=FONT_LABEL, fontweight='bold')
        ax.set_ylabel('Petal Length', fontsize=FONT_LABEL, fontweight='bold')
        ax.set_zlabel('Probability (%)', fontsize=FONT_LABEL, fontweight='bold')
        
        # 设置 z 轴范围以显示投影 (从 -100 到 概率最大值)
        ax.set_zlim(-100, max(100, probs.max() + 10))
        
        ax.view_init(elev=28, azim=135)
        ax.tick_params(labelsize=9)
    
    fig.suptitle('Task 3: 3D Probability Contour Maps',
                fontsize=FONT_TITLE_MAIN, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.35, top=0.88)
    
    save_figure('Task3_Probability_Contours', fig)
    plt.close(fig)


# ==========================================================
# TASK 4: 3D Boundary Surface + 3D Contour Map (2 figures)
# ==========================================================
def task4_multiclass():
    print("\n" + "="*60)
    print("TASK 4: 3-Class Analysis")
    print("="*60)
    
    iris = load_iris()
    X_full = iris.data
    y = iris.target
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    
    X_3d_scaled = X_scaled[:, [1, 2, 3]]  # Sepal Width, Petal Length, Petal Width
    
    # Train multi-class classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_3d_scaled, y)
    
    train_acc = clf.score(X_3d_scaled, y)
    print(f"  Accuracy: {train_acc*100:.1f}%")
    
    # === 4-A: 3D Decision Boundary Surface ===
    print("\n  [4-A] Rendering 3D decision boundary surface...")
    
    fig = plt.figure(figsize=(12, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F8FBFF')  # 冷色系背景
    
    # Create grid with sufficient density
    resolution = 30
    mins = X_3d_scaled.min(axis=0) - 0.5
    maxs = X_3d_scaled.max(axis=0) + 0.5
    
    xx = np.linspace(mins[0], maxs[0], resolution)
    yy = np.linspace(mins[1], maxs[1], resolution)
    zz = np.linspace(mins[2], maxs[2], resolution)
    
    XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing='ij')
    grid_points = np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()]
    
    # Predict on grid
    predictions = clf.predict(grid_points)
    vol = predictions.reshape(XX.shape).astype(float)
    
    # Smooth for better surface
    vol_smooth = gaussian_filter(vol, sigma=1.0)
    
    # Marching cubes to extract surfaces
    for class_idx in range(3):
        threshold = class_idx + 0.5
        try:
            verts, faces, _, _ = measure.marching_cubes(vol_smooth, level=threshold)
            
            if len(verts) > 0:
                # Map verts back to original coordinate space
                verts_mapped = np.zeros_like(verts)
                for dim in range(3):
                    verts_mapped[:, dim] = mins[dim] + verts[:, dim] * (maxs[dim] - mins[dim]) / (resolution - 1)
                
                # Create mesh for this class
                mesh = Poly3DCollection(verts_mapped[faces], alpha=0.25, edgecolor='none')
                mesh.set_facecolor(COLORS[class_idx])
                ax.add_collection3d(mesh)
        except Exception as e:
            print(f"    Warning: Could not extract surface for class {class_idx}")
    
    # Plot data points with cool colors
    for class_idx in range(3):
        pts = X_3d_scaled[y == class_idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                  c=COLORS_COOL[class_idx], s=80, edgecolor='black',
                  linewidth=1.2, label=CLASS_NAMES[class_idx],
                  alpha=0.85, zorder=10)
    
    ax.set_title(f'Task 4-A: 3D Decision Boundary Surface\nAccuracy: {train_acc*100:.1f}%',
                fontsize=FONT_TITLE_MAIN, fontweight='bold', pad=20)
    ax.set_xlabel('Sepal Width', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Petal Length', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_zlabel('Petal Width', fontsize=FONT_LABEL, fontweight='bold')
    
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95, edgecolor='black')
    ax.view_init(elev=25, azim=45)
    ax.tick_params(labelsize=9)
    
    save_figure('Task4_Decision_Surface', fig, save_pdf=False)
    plt.close(fig)
    
    # === 4-B: 3D Contour Map (probability visualization) ===
    print("\n  [4-B] Rendering 3D contour map...")
    
    # Train SVM for probability estimation (more reliable than LR proba)
    svm_multi = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
    svm_multi.fit(X_3d_scaled, y)
    
    # Create fine mesh for class 1 (Versicolor) probability
    RES = 100
    pad = 1.5
    x_min, x_max = X_3d_scaled[:, 0].min() - pad, X_3d_scaled[:, 0].max() + pad
    y_min, y_max = X_3d_scaled[:, 1].min() - pad, X_3d_scaled[:, 1].max() + pad
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, RES),
                         np.linspace(y_min, y_max, RES))
    
    z_fixed = X_3d_scaled[:, 2].mean()
    grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z_fixed)]
    probs = svm_multi.predict_proba(grid)
    Z = probs[:, 1].reshape(xx.shape) * 100  # Versicolor probability
    
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F8FBFF')  # 冷色系背景
    
    # === High-contrast probability map styling ===
    z_base = np.zeros_like(xx)
    
    # 1. High-contrast baseline plane: DeepSkyBlue with opacity
    ax.plot_surface(xx, yy, z_base, color='deepskyblue', alpha=0.4, shade=False, zorder=1)
    
    # 2. White grid overlay on baseline
    ax.plot_wireframe(xx, yy, z_base, rstride=10, cstride=10, 
                     color='white', linewidth=0.5, alpha=0.6, zorder=2)
    
    # 3. Bold black decision line (50% probability threshold)
    ax.contour(xx, yy, Z, levels=[50], colors='black', linewidths=4, 
              zdir='z', offset=0, zorder=10)
    
    # 4. Probability surface with RdYlBu colormap (对比度强)
    cmap_prob = plt.cm.RdYlBu_r  # 红黄蓝对比colormap（红=高概率，蓝=低概率）
    ax.plot_surface(xx, yy, Z, cmap=cmap_prob, alpha=0.75, 
                   shade=True, edgecolor='none', zorder=5)
    
    # 5. Dark skeleton wireframe
    ax.plot_wireframe(xx, yy, Z, rstride=6, cstride=6, 
                     color='midnightblue', linewidth=0.6, alpha=0.35, zorder=5)
    
    # 6. 三个坐标面投影
    z_min = -100
    x_min = xx.min()
    y_min = yy.min()
    
    # 底面投影 (Z = -100)
    ax.contourf(xx, yy, Z, zdir='z', offset=z_min, cmap=cmap_prob, alpha=0.4, levels=12)
    
    # 左侧投影 (X = x_min)
    ax.contourf(xx, yy, Z, zdir='x', offset=x_min, cmap=cmap_prob, alpha=0.4, levels=12)
    
    # 正面投影 (Y = y_min)
    ax.contourf(xx, yy, Z, zdir='y', offset=y_min, cmap=cmap_prob, alpha=0.4, levels=12)
    
    # Project data points with cool colors
    for class_idx in range(3):
        pts = X_3d_scaled[y == class_idx]
        ax.scatter(pts[:, 0], pts[:, 1], np.zeros(len(pts)),
                  c=COLORS_COOL[class_idx], s=80, edgecolor='black', linewidth=1.2,
                  marker=MARKERS[class_idx], label=CLASS_NAMES[class_idx],
                  alpha=0.85, zorder=10)
    
    ax.set_title('Task 4-B: 3D Probability Contour Map',
                fontsize=FONT_TITLE_MAIN, fontweight='bold', pad=20)
    ax.set_xlabel('Sepal Width', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_ylabel('Petal Length', fontsize=FONT_LABEL, fontweight='bold')
    ax.set_zlabel('Probability (%)', fontsize=FONT_LABEL, fontweight='bold')
    
    # 设置 z 轴范围以显示投影 (从 -100 到概率最大值)
    ax.set_zlim(-100, max(100, Z.max() + 10))
    
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95, edgecolor='black')
    ax.view_init(elev=28, azim=135)
    ax.tick_params(labelsize=9)
    
    save_figure('Task4_Contour_Map', fig, save_pdf=False)
    plt.close(fig)


# ==========================================================
# Main Execution
# ==========================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("IRIS CLASSIFICATION VISUALIZATION (FINAL VERSION)")
    print("="*60)
    
    task1_2d_decision_boundary()
    task2_3d_binary_surfaces()
    task3_3d_probability_contours()
    task4_multiclass()
    
    print("\n" + "="*60)
    print("✓ All tasks completed! Files saved to: " + OUTPUT_DIR)
    print("="*60 + "\n")
