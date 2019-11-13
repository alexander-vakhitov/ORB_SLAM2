import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def draw_ellipse(ax, S2d, c2d, d, col=(0,0,1), alpha=1.0):
    u,s,v = np.linalg.svd(S2d)
    w = 2*np.sqrt(d / s[0])
    h = 2*np.sqrt(d / s[1])
    phi = np.arctan2(u[1,0], u[0,0])
    e3d = Ellipse(xy=c2d, angle = phi * 180.0 / np.pi, height=h, width=w, edgecolor=col, facecolor='none', alpha=alpha)

    ec = Ellipse(xy=c2d, angle=phi * 180.0 / np.pi, height=1, width=1, color=col,
                  alpha=1.0)

    ax.add_artist(e3d)
    ax.add_artist(ec)
    return ec


def project_ellipse(Sigma, K, c, d = 1.0):
    C = c.reshape((3,1)).dot(c.reshape((1,3)))
    SigmaProj = (c.reshape((1,3)).dot(Sigma).dot(c.reshape((3,1))) - d)*Sigma -Sigma.dot(C).dot(Sigma)
    Ki = np.linalg.inv(K)
    SigmaProj = Ki.transpose().dot(SigmaProj).dot(Ki)
    S2d = SigmaProj[0:2,0:2]
    c2d = -np.linalg.inv(S2d).dot(SigmaProj[0:2,2])
    d2d = -SigmaProj[0:2,2].dot(c2d)-SigmaProj[2,2]
    return S2d, c2d, d2d

def ellipse_proj_to_euc(Sigma):
    S2d = Sigma[0:2, 0:2]
    c2d = -Sigma[0:2, 0:2] @ Sigma[0:2, 2]
    d2d = -Sigma[2, 2] + Sigma[0:2, 2].T @ Sigma[0:2, 0:2] @ Sigma[0:2, 2]
    return S2d, c2d, d2d

def project_ellipse_2(Sigma, c, d):
    pass

def find_2d_ellipse_projection(A, c, d, X_true=None):
#we project along x
#we parametrize a point as (x, y)
#we express y from -c' A ( [x, y]' - c ) = d
#y = alpha * x + b
    is_xy_switch = False
    a1 = A[:, 0]
    a2 = A[:, 1]
    if np.abs(np.dot(c, a2)) < 1e-6:
        # switch x and y
        is_xy_switch = True
        cm = c[0]
        c[0] = c[1]
        c[1] = cm
        # am = np.copy(A[:, 0])
        # A[:, 0] = A[:, 1]
        # A[:, 1] = am
        a1 = A[:, 0]
        a2 = A[:, 1]
        if X_true is not None:
            xm = X_true[0]
            X_true[0] = X_true[1]
            X_true[1] = xm
            print(A)
            print(c)
            print('eq test')
            print((X_true-c).T @ A @ (X_true-c) - d)
            print('touch test')
            print((-c).T @ A @ (X_true - c) - d)
            print((-c).T @ (a1 * X_true[0] + a2 * X_true[1] - A @ c) - d)
            print(-np.dot(c, a1) * X_true[0] -np.dot(c, a2) * X_true[1] + c.T @ A @ c - d)

    alpha = -np.dot(c, a1) / np.dot(c, a2)
    b = (c.T @ A @ c - d) / (np.dot(c, a2))
    if X_true is not None:
        print('touch test')
        print(alpha*X_true[0] + b - X_true[1])
#ellipse equation ([x,y]' - c)' A ([x,y] - c) = d
#let t = x - c[0]
    k = alpha * c[0] + b - c[1]
#equation reduces to
#t^2 * A[0,0] + 2 t (alpha * t + k) A[0,1] + (alpha * t + k)^2 * A[1,1] - d = 0
#which is
#(A[0,0] + 2 alpha A[0,1] + alpha^2 A[1,1]) t^2 +
#(2 k A[0,1] + 2 alpha k A[1,1]) t +
#k^2 A[1,1] - d = 0
    a_coeff = A[0,0] + 2 * alpha * A[0,1] + alpha ** 2 * A[1,1]
    b_coeff = k * A[0,1] + k * alpha * A[1,1]
    c_coeff = k ** 2 *A[1,1] - d
    D4 = b_coeff ** 2 - a_coeff * c_coeff

    t_1 = (b_coeff + np.sqrt(D4)) / a_coeff
    t_2 = (b_coeff - np.sqrt(D4)) / a_coeff
    x_1 = t_1 + c[0]
    x_2 = t_2 + c[0]
    X_1 = np.asarray([x_1, alpha * x_1 + b])
    X_2 = np.asarray([x_2, alpha * x_2 + b])
#test
    def fun(X):
        return (X-c).T @ A @ (X-c) - d
    def fun_2(X):
        return (X ).T @ A @ (X - c)

    print(X_1)
    print(fun(X_1))
    print(fun_2(X_1))
    print(X_2)
    print(fun(X_2))
    print(fun_2(X_2))











def findHomography(X_from, X_to):
    A = np.zeros((12, 13))
    for i in range(0, 4):
        for j in range(0, 3):
            A[3*i+j, 3*j:3*(j+1)] = X_from[0:3, i]
            A[3*i+j, 9+i] = -X_to[j, i]

    u,s,v = np.linalg.svd(A)
    p = v[12, :]
    print(np.linalg.norm(A.dot(p)))
    H = np.reshape(p[0:9],(3,3))
    for j in range(0, 4):
        x0h = H.dot(X_from[:, j])
        x0h = x0h[0:2]/x0h[2]
        xtoh = X_to[0:2, j]/X_to[2,j]
        print(np.linalg.norm(x0h - xtoh))
    print('end')
    return H




def project_uncertainty_slice(Sigma, K, c, d=1.0):
    mini = np.argmin(np.abs(c))
    dir0 = np.zeros(3)
    dir0[mini] = 1.0
    d1 = np.cross(dir0, c)
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.cross(c, d1)
    D = np.zeros((2,3))
    D[0, :] = d1
    D[1, :] = d2
    SigmaSlice = D.dot(Sigma).dot(D.transpose())
    X_loc = np.ones((3,4))
    X_loc[0:2, 0] = 0
    X_loc[0:2, 1] = np.asarray([1, 0])
    X_loc[0:2, 2] = np.asarray([0, 1])
    X_loc[0:2, 3] = np.asarray([1, 1])

    X = np.ones((3,4))
    X[0:3, 0] = K.dot(c)
    X[0:3, 1] = K.dot(c+d1)
    X[0:3, 2] = K.dot(c + d2)
    X[0:3, 3] = K.dot(c + d2 + d1)
    H = findHomography(X_loc, X)
    Hi = np.linalg.inv(H)
    SigmaSliceH = -np.eye(3)
    SigmaSliceH[0:2,0:2] = SigmaSlice[0:2,0:2]
    SigmaH = Hi.transpose().dot(SigmaSliceH).dot(Hi)
    SigmaH = -SigmaH/SigmaH[2,2]
    S2d = SigmaH[0:2,0:2]
    c2d = -np.linalg.inv(S2d).dot(SigmaH[0:2,2])
    d2d = -SigmaH[2,2] + c2d.transpose().dot(S2d).dot(c2d)

    S2d = S2d/d2d
    ch = K.dot(c)
    ch2 = H.dot([0,0,1])
    print(ch2/ch2[2])
    print(ch[0:2]/ch[2])
    print(c2d)

    print('-')

    return S2d, c2d, d2d



def test_draw_ellipse():
    ax = plt.gca()
    S2d = np.diag([1, 10])
    b = np.pi/4
    R = np.asarray([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
    S2d = R.dot(S2d).dot(R.transpose())
    c2d = np.ones(2)
    c2d[0] = 0.5
    c2d[1] = 0.8
    d = 1.0
    draw_ellipse(ax, S2d, c2d, d)
    plt.show()



def test_project_ellipse():
    S3d = np.eye(3)
    # S3d[0,0] = 10
    # S3d[2,2] = 100
    c3d = np.zeros(3)
    c3d[2] = 1.1
    c3d[1] = 2
    S2d, c2d, d2d = project_ellipse(S3d, np.eye(3), c3d)
    print('2d center is '+str(c2d))
    print('projected 3d center is '+str(c3d[0:2]/c3d[2]))
    ax = plt.gca()
    ax.figure.set_size_inches(5, 5)
    draw_ellipse(ax, S2d, c2d, d2d)
    ax.set_xlim(left=-3,right=3)
    ax.set_ylim(bottom=-3,top=15)
    plt.show()

def test_project_ellipse_real():
    S3d = np.asarray([[0.28917735,  0.07744052, - 0.96230381],
     [0.07744052,  0.02103052, -0.25792143],
    [-0.96230381, -0.25792143,    3.2050047]])
    c3d = np.asarray([-3.31214009, -0.87968205, 10.78984004])
    K = np.asarray([[718.856,   0.,    607.193],
     [0.,    718.856, 185.216],
    [0.,   0.,   1.]])

    u,s,v = np.linalg.svd(S3d)
    print(s)

    S2d, c2d, d2d = project_ellipse(S3d, K, c3d)

    u, s, v = np.linalg.svd(S2d)
    print(s)

# test_project_ellipse()

# test_draw_ellipse()

# test_project_ellipse_real()
def test1():
    print('correct answer')
    beta = np.arccos(0.5)
    y_t = np.sin(beta)
    x_t = np.cos(np.pi/2-beta) * np.sqrt(3)
    X_true = np.asarray([x_t, y_t])
    print(X_true)
    A = np.eye(2)
    c = np.asarray([2, 0])
    d = 1
    print('touch')
    print(-c.T @ A @ (X_true - c) - d)
    find_2d_ellipse_projection(A, c, d, X_true)

def test2():
    A = np.asarray([[2,0], [0,1]])
    c = np.asarray([2, 0])
    d = 1
    find_2d_ellipse_projection(A, c, d)

def test2():
    A = np.asarray([[2,0], [0,1]])
    c = np.asarray([2, 0])
    d = 1
    find_2d_ellipse_projection(A, c, d)

# test1()
test2()