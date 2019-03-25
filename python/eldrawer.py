import numpy as np

def define_ellipsoid(X0, Sigma):
    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)

    U, scale, rotation = np.linalg.svd(Sigma)
    scale = np.sqrt(scale)
    # v_min = np.min(scale)
    # v_max = np.max(scale)
    v_m = np.mean(scale)
    for i in range(0,3):
        scale[i] = np.min([np.max([0.2*v_m, scale[i]]), 5*v_m])

    x = scale[0]*np.outer(np.cos(u), np.sin(v))
    y = scale[1]*np.outer(np.sin(u), np.sin(v))
    z = scale[2]*np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + X0
    return x,y,z


