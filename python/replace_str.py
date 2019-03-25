def replace_file_cpp(f, out_path, array_lbl):
    f_in = open(f, 'r')
    f_out = open(out_path, 'w')
    letters = ['a', 'b', 'c', 'd']
    cnt = 0
    for line in f_in:
        for i in range(1, 10):
            line = line.replace('_'+str(i), ','+str(i-1)+')')
        for i in range(1, 10):
            line = line.replace('A'+str(i) + ',', 'A('+str(i-1)+',')
        for l in letters:
            line = line.replace(l + '^2' , l+'*'+l)
        line = line.replace('(a*a + b*b + c*c + 1)^2', '(a*a + b*b + c*c + 1)*(a*a + b*b + c*c + 1)')
        line = line.replace('(a*a + b*b + c*c + 1)^3', '(a*a + b*b + c*c + 1)*(a*a + b*b + c*c + 1)*(a*a + b*b + c*c + 1)')
        line = array_lbl+'['+str(cnt)+'] = '+line
        cnt += 1
        f_out.write(line)

def replace_file_matlab(f, out_path):
    f_in = open(f, 'r')
    f_out = open(out_path, 'w')
    for line in f_in:
        for i in range(1, 10):
            line = line.replace('_'+str(i), ','+str(i)+')')
        for i in range(1, 10):
            line = line.replace('A'+str(i) + ',', 'A('+str(i)+',')
        f_out.write(line+';'+'\n')

# replace_file_matlab('/home/alexander/materials/pnp3d/pnp3d/PnP3D_Toolbox/g.txt', '/home/alexander/materials/pnp3d/pnp3d/PnP3D_Toolbox/gm.txt')
replace_file_cpp('/home/alexander/materials/pnp3d/pnp3d/PnP3D_Toolbox/H.txt', '/home/alexander/materials/pnp3d/pnp3d/PnP3D_Toolbox/Hc.txt', 'H')
replace_file_cpp('/home/alexander/materials/pnp3d/pnp3d/PnP3D_Toolbox/g.txt', '/home/alexander/materials/pnp3d/pnp3d/PnP3D_Toolbox/gc.txt', 'g')
