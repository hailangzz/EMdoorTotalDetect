import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# ======================
# 1. 标定数据
# ======================
x = np.array([0.07,0.10,0.20,0.30,0.40,0.49,0.56,0.645,0.73,0.795,0.875,0.935,1.015,1.075,1.135,0.34,0.47,0.62,0.75,0.85,0.93,1.015,1.06,1.13,1.21,1.26,1.345,1.40,1.51,1.565], dtype=np.float32)
y = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0], dtype=np.float32)

X = x.reshape(-1, 1)

# ======================
# 2. 二次多项式 + Ridge
# ======================
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

model = Ridge(alpha=0.1, fit_intercept=False)
model.fit(X_poly, y)

# 提取参数（用于部署）
w0, w1, w2 = model.coef_
print(f"y = {w2:.4f} x^2 + {w1:.4f} x + {w0:.4f}")

# ======================
# 3. 生成拟合曲线
# ======================
x_fit = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
x_fit_poly = poly.transform(x_fit)
y_fit = model.predict(x_fit_poly)

# ======================
# 4. 绘图
# ======================
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="Ground Truth", s=60)
plt.plot(x_fit, y_fit, label="Ridge Polynomial Fit", linewidth=2)

plt.xlabel("Detection Value")
plt.ylabel("Real Distance")
plt.title("Quadratic Ridge Regression")
plt.legend()
plt.grid(True)

# ======================
# 5. 保存图片
# ======================
plt.savefig("ridge_quadratic_fit.png", dpi=300, bbox_inches="tight")
plt.show()
