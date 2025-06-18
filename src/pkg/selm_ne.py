# SELM Non-equilibrium Package 

# -- imports
import numpy as np
import pickle;
import time; 
import os; 
import sys;
import shutil;

import argparse; 

import vtk

# -- functions
def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);   

def get_mesh_xx(params):
  num_mesh_x,num_mesh_y,deltaX = \
    tuple(map(params.get,['num_mesh_x','num_mesh_y','deltaX']));
  ss = np.linspace(0,num_mesh_x-1,num_mesh_x)*deltaX;
  tt = np.array([0.0]);
  xx = [ss,ss,tt];
  return xx; 

def get_mesh_yy(params):
  xx = get_mesh_xx(params);
  yyy = np.meshgrid(xx[0],xx[1]);
  y1 = yyy[0].flatten();  y2 = yyy[1].flatten();
  return y1,y2; 

def get_mesh_coord(params):
  xx = get_mesh_xx(params); 
  yy = np.meshgrid(xx[0],xx[1]);
  x1 = yy[0].flatten(); x2 = yy[1].flatten();
  deltaX = params['deltaX'];
  Lx = params['num_mesh_x']*deltaX;
  Ly = params['num_mesh_y']*deltaX;
  return x1,x2,Lx,Ly;

def conv_2d_vec_3d(X,num_dim):
  num_points = X.shape[0]//num_dim;
  ambient_num_dim = 3;
  XX = np.zeros((num_points,ambient_num_dim));
  if num_dim == 2:
    XX[:,0:num_dim] = X.reshape(num_points,num_dim);
    XX[:,2] = 0.0;
  elif num_dim == 3:
    XX = X.reshape(num_points,num_dim);
  else:
    raise Exception("Expecting particles to have num_dim = 2 or 3.");

  return XX; 

def create_fdata(aa,name='a'):
  if aa.ndim == 1:
    num_components = 1;
  else:
    num_components = aa.shape[1];

  ff = {'field_name':name,
        'NumberOfComponents':num_components,
        'field_values':aa};
 
  return ff;  

def add_fdata(flist,aa,name='a'):
  fdata = create_fdata(aa,name);
  flist.append(fdata);
  return flist;

def write_vtr_data(filename,xx,field_list,flag_verbose=1):

  vtr_grid = vtk.vtkRectilinearGrid();

  vtr_grid.SetDimensions(xx[0].shape[0],
                         xx[1].shape[0],
                         xx[2].shape[0]);

  num_points = 1;
  for d in range(0,3):
    num_points *= xx[d].shape[0];
     
  x1Array = vtk.vtkDoubleArray();
  x1Array.SetName('x1');
  for x1 in xx[0]: x1Array.InsertNextValue(x1);

  x2Array = vtk.vtkDoubleArray();
  x2Array.SetName('x2');
  for x2 in xx[1]: x2Array.InsertNextValue(x2);

  x3Array = vtk.vtkDoubleArray();
  x3Array.SetName('x3');
  for x3 in xx[2]: x3Array.InsertNextValue(x3);

  vtr_grid.SetXCoordinates(x1Array);
  vtr_grid.SetYCoordinates(x2Array);
  vtr_grid.SetZCoordinates(x3Array);

  # -- setup data arrays
  num_fields = len(field_list);
  for field_data in field_list:
    field_name,field_values,NumberOfComponents,data_org = \
      tuple(map(field_data.get,['field_name','field_values',
                                'NumberOfComponents','data_org']));
    if data_org is None:
      data_org = 'point_data'; # default 

    # special case of 2 components we convert to 3
    # assuming the z-direction is zero.
    if (NumberOfComponents == 2):
      ambient_num_dim = 3; ss = field_values.shape; 
      field_values = \
        conv_2d_vec_3d(field_values.flatten(),NumberOfComponents).reshape(ss[0],3);
      NumberOfComponents = 3;

    if (NumberOfComponents == 1):
      # setup the scalar_ field data
      num_components = 1;
      f_data = vtk.vtkFloatArray();
      f_data.SetName(field_name);
      f_data.SetNumberOfComponents(num_components);
      nn = field_values.shape[0];
      f_data.SetNumberOfTuples(nn);
      for I in np.arange(0,nn):
        f_data.SetValue(I,field_values[I]);

      if data_org == 'point_data':
        vtr_grid.GetPointData().AddArray(f_data);
      elif data_org == 'cell_data':
        vtr_grid.GetCellData().AddArray(f_data);
      else: 
        raise Exception("Not recognized, data_org = " + data_org);

    elif (NumberOfComponents == 3): 
      # setup the vector_ field data
      ambient_num_dim = num_components = 3;
      f_data = vtk.vtkFloatArray();
      f_data.SetName(field_name);
      f_data.SetNumberOfComponents(num_components);
      nn = field_values.shape[0];
      f_data.SetNumberOfTuples(nn);
      nn = field_values.shape[0];
      for I in np.arange(0,nn):
        for d in range(0,ambient_num_dim):
          f_data.SetValue(I*ambient_num_dim + d,field_values[I,d]);

      if data_org == 'point_data':
        vtr_grid.GetPointData().AddArray(f_data);
      elif data_org == 'cell_data':
        vtr_grid.GetCellData().AddArray(f_data);
      else: 
        raise Exception("Not recognized, data_org = " + data_org);
    else:
      ss = "";
      ss += "NumberOfComponents invalid. \n";
      ss += "NumberOfComponents = " + str(NumberOfComponents);
      raise Exception(ss);

  # write the structured grid to XML file
  vtr_writer = vtk.vtkXMLRectilinearGridWriter();
  vtr_writer.SetFileName(filename);
  vtr_writer.SetInputData(vtr_grid);
  vtr_writer.SetCompressorTypeToNone(); # help ensure ascii output (as opposed to binary)
  vtr_writer.SetDataModeToAscii(); # help ensure ascii output (as opposed to binary)
  vtr_writer.Write();

def write_vtp_data(vtp_filename,points,field_list,flag_verbose=1):
  # output a VTP file with the fields

  vtp_data = vtk.vtkPolyData();

  # setup the points data
  vtp_points = vtk.vtkPoints();
  ambient_num_dim = points.shape[1]; 
  num_points = points.shape[0];
  for I in range(num_points):
    vtp_points.InsertNextPoint(points[I,0],points[I,1],points[I,2]);
  
  vtp_data.SetPoints(vtp_points);

  # -- setup data arrays
  num_fields = len(field_list);
  if flag_verbose >= 2:
    print("num_fields = " + str(num_fields));
  for field_data in field_list:
    if flag_verbose >= 2:
      print("field_data = " + str(field_data));
    field_name = field_data['field_name'];
    field_values = field_data['field_values'];
    NumberOfComponents = field_data['NumberOfComponents'];

    if (NumberOfComponents == 1):
      N_list = field_values.shape[0];
      atz_data_phi = vtk.vtkDoubleArray();
      atz_data_phi.SetNumberOfComponents(1);
      atz_data_phi.SetName(field_name);
      atz_data_phi.SetNumberOfTuples(N_list);
      for I in np.arange(0,N_list):
        atz_data_phi.SetValue(I,field_values[I]);
      vtp_data.GetPointData().AddArray(atz_data_phi);
    elif (NumberOfComponents == 3): 
      N_list = field_values.shape[0];
      atz_data_V = vtk.vtkDoubleArray();
      atz_data_V.SetNumberOfComponents(3);
      atz_data_V.SetName(field_name);
      atz_data_V.SetNumberOfTuples(N_list);
      for I in np.arange(0,N_list):
        atz_data_V.SetValue(I*ambient_num_dim + 0,field_values[I,0]);
        atz_data_V.SetValue(I*ambient_num_dim + 1,field_values[I,1]);
        atz_data_V.SetValue(I*ambient_num_dim + 2,field_values[I,2]);
      vtp_data.GetPointData().AddArray(atz_data_V);
    else:
      ss = ""; ss += "NumberOfComponents invalid (needs to be 1 or 3). \n";
      ss += "NumberOfComponents = " + str(NumberOfComponents) + "\n";
      ss += "field_name = " + field_name + "\n";
      raise Exception(ss);

  # write the unstructured grid to XML file
  writer_vtp = vtk.vtkXMLPolyDataWriter();
  writer_vtp.SetFileName(vtp_filename);
  writer_vtp.SetInputData(vtp_data);
  writer_vtp.SetCompressorTypeToNone(); # help ensure ascii output (as opposed to binary)
  writer_vtp.SetDataModeToAscii(); # help ensure ascii output (as opposed to binary)
  writer_vtp.Write();   

def write_array_vtr(a,params,name=None,filename=None):
  if name is None:
    name = 'a';

  if filename is None: 
    filename = params['base_dir'] + '/debug/%s.vtr'%name;

  if a.ndim > 1:
    num_components = a.shape[1];
  else:
    num_components = 1;

  xx = get_mesh_xx(params);
  ff = {};
  ff['NumberOfComponents'] = num_components;
  ff['field_values'] = a;
  ff['field_name'] = 'a';
  fff = [ff];
  write_vtr_data(filename,xx,fff);

def get_comp(Y,i1_str,i2_str,Y_I):
  return Y[Y_I[i1_str]:Y_I[i2_str]];

def set_comp(Y,i1_str,i2_str,Y_I,val):
  Y[Y_I[i1_str]:Y_I[i2_str]] = val;

def add_in_components(Y_out,a_out,I_out,I_local_out): 
  ii1_list = I_out['i1']; ii2_list = I_out['i2']; 
  i1_list = I_local_out['i1']; i2_list = I_local_out['i2']; 
  for k in range(0,len(i1_list)): 
    i1 = i1_list[k]; i2 = i2_list[k];
    ii1 = ii1_list[k]; ii2 = ii2_list[k];
    Y_out[ii1:ii2] += a_out[i1:i2];

def add_in_matrix_entries(A,B,I_out,I_in,I_local_out,I_local_in): 
  ii1_list = I_out['i1']; ii2_list = I_out['i2']; 
  jj1_list = I_in['i1']; jj2_list = I_in['i2']; 
  i1_list = I_local_out['i1']; i2_list = I_local_out['i2']; 
  j1_list = I_local_in['i1']; j2_list = I_local_in['i2']; 
  for k in range(0,len(i1_list)): 
    i1 = i1_list[k]; i2 = i2_list[k];
    j1 = j1_list[k]; j2 = j2_list[k];
    ii1 = ii1_list[k]; ii2 = ii2_list[k];
    jj1 = jj1_list[k]; jj2 = jj2_list[k];
    A[ii1:ii2,jj1:jj2] += B[i1:i2,j1:j2];

def get_parts_I(params):
  return params['Y_I'];

def map_particle_periodic(Y,params):
  Y_I = params['Y_I']; num_dim = params['num_dim'];
  num_mesh_x,num_mesh_y,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','deltaX']));
  Lx = num_mesh_x*deltaX; Ly = num_mesh_y*deltaX;

  I1_particle_q,I2_particle_q = tuple(map(
    Y_I.get,['I1_particle_q','I2_particle_q']));

  # hard-coded num_dim = 2
  if num_dim != 2:
    raise Exception("implemented currently only for num_dim == 2");
 
  Y[I1_particle_q + 0] = np.mod(Y[I1_particle_q + 0],Lx);
  Y[I1_particle_q + 1] = np.mod(Y[I1_particle_q + 1],Ly);

def compute_matrix_vec_div(Y,params,extras=None):
  """ Divergence acting on vectors $v$ to produce scalar $a$, 
      $a = div(v),\; a = \partial_{j} v_{j}$.
  """
  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX \
    = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));  

  if num_dim != 2:
    raise Exception("assumes num_dim = 2, input gave num_dim = " + str(num_dim));

  if extras is not None:
    D,flag_save = tuple(map(extras.get,['D','flag_save']));
    if flag_save is None:
      flag_save = False; 
  else:
    D = None;
    flag_save = False; 

  if D is not None: # no need to re-compute, just return it
    return D;

  # div: maps R^{dxd} -> R^d, (maps tensor to vector)
  D = np.zeros((num_mesh_x*num_mesh_y,num_mesh_x*num_mesh_y*num_dim));
  
  # @ optimize 
  vec_I = np.zeros(num_dim,dtype=int); vec_num_mesh = np.zeros(num_dim,dtype=int);
  vec_num_mesh[0] = num_mesh_x; vec_num_mesh[1] = num_mesh_y;
  J0_mesh_iim1 = np.zeros(num_dim,dtype=int); J0_mesh_iip1 = np.zeros(num_dim,dtype=int); 
  for j in range(0,num_mesh_y): 
    for i in range(0,num_mesh_x):
      vec_I[0] = i; vec_I[1] = j; 
      vec_Im1 = np.zeros(vec_I.shape,dtype=int); vec_Ip1 = np.zeros(vec_I.shape,dtype=int); # same shape arrays
      I0_mesh = vec_I[1]*num_mesh_x + vec_I[0] + 0;
      for d in range(0,num_dim):
        vec_Im1[:] = vec_I[:]; vec_Ip1[:] = vec_I[:]; # copy index then perturb it only in index d 
        iim1 = vec_I[d] - 1; iip1 = vec_I[d] + 1;
        if iim1 < 0: 
          iim1 = vec_num_mesh[d] + iim1; # periodic
        if iip1 >= vec_num_mesh[d]: 
          iip1 = iip1 - vec_num_mesh[d]; # periodic
        vec_Im1[d] = iim1; vec_Ip1[d] = iip1; 
        
        J0_mesh_iim1[d] = vec_Im1[1]*num_mesh_x*num_dim + vec_Im1[0]*num_dim + 0;
        J0_mesh_iip1[d] = vec_Ip1[1]*num_mesh_x*num_dim + vec_Ip1[0]*num_dim + 0;

      for b in range(0,num_dim):
        D[I0_mesh,J0_mesh_iip1[b] + b] += 1.0/(2.0*deltaX);
        D[I0_mesh,J0_mesh_iim1[b] + b] += -1.0/(2.0*deltaX); 

  if flag_save: 
    extras['D'] = D;

  return D;

def compute_matrix_vec_grad(Y,params,extras=None):
  """ Gradient acting on scalar fields, such as $\nabla f$. """

  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX \
    = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));  
  num_dim_sq = num_dim*num_dim; # tensor dimension 

  if num_dim != 2:
    raise Exception("assumes num_dim = 2, input gave num_dim = " + str(num_dim));

  if extras is not None:
    G,flag_save = tuple(map(extras.get,['G','flag_save']));
    if flag_save is None:
      flag_save = False; 
  else:
    G = None;
    flag_save = False; 

  if G is not None: # no need to re-compute, just return it
    return G;

  # grad: maps R^{d} -> R^{dxd}, (maps vector to tensor)
  G = np.zeros((num_mesh_x*num_mesh_y*num_dim,num_mesh_x*num_mesh_y));
  
  # @ optimize 
  vec_I = np.zeros(num_dim,dtype=int); vec_num_mesh = np.zeros(num_dim,dtype=int);
  vec_num_mesh[0] = num_mesh_x; vec_num_mesh[1] = num_mesh_y;
  J0_mesh_iim1 = np.zeros(num_dim,dtype=int); J0_mesh_iip1 = np.zeros(num_dim,dtype=int); 
  for j in range(0,num_mesh_y): 
    for i in range(0,num_mesh_x):
      vec_I[0] = i; vec_I[1] = j; 
      vec_Im1 = vec_I + 0; vec_Ip1 = vec_I + 0; # make copies
      I0_mesh = vec_I[1]*num_mesh_x*num_dim + vec_I[0]*num_dim + 0;
      for d in range(0,num_dim):
        vec_Im1[:] = vec_I[:]; vec_Ip1[:] = vec_I[:]; # copy index then perturb it only in index d 
        iim1 = vec_I[d] - 1; iip1 = vec_I[d] + 1;
        if iim1 < 0: 
          iim1 = vec_num_mesh[d] + iim1; # periodic
        if iip1 >= vec_num_mesh[d]: 
          iip1 = iip1 - vec_num_mesh[d]; # periodic
        vec_Im1[d] = iim1; vec_Ip1[d] = iip1; 
        
        J0_mesh_iim1[d] = vec_Im1[1]*num_mesh_x + vec_Im1[0] + 0;
        J0_mesh_iip1[d] = vec_Ip1[1]*num_mesh_x + vec_Ip1[0] + 0;

      for b in range(0,num_dim):
        G[I0_mesh + b,J0_mesh_iip1[b]] += 1.0/(2.0*deltaX);
        G[I0_mesh + b,J0_mesh_iim1[b]] += -1.0/(2.0*deltaX); 

  if flag_save:
    extras['G'] = G;

  return G;

def peskin_delta(rr,extras=None):
 
  tt = type(rr);
  if tt == np.ndarray:
    rrr = rr; 
    w = 0*rr; 
  elif tt == np.float64 or tt == float:
    rrr = np.array([rr]);
    w = np.zeros(rrr.shape);
  elif tt == int: 
    rrr = np.array([float(rr)]);
    w = np.zeros(rrr.shape);
  else:
    raise Exception('Input type not able to be handled.');

  r = rrr; 
  I0 = np.nonzero(r <= -2);
  I1 = np.nonzero(-2 < r and r < -1);
  I2 = np.nonzero(-1 <= r and r < 0);
  I3 = np.nonzero(0 <= r and r < 1);
  I4 = np.nonzero(1 <= r and r < 2);
  I5 = np.nonzero(r >= 2);

  # w[I0] = 0.0;
  if len(I1) > 0:
    r = rrr[I1]; w[I1] = (1.0/8.0)*(5.0 + 2.0*r - np.sqrt(-7.0 - 12.0*r - 4.0*r*r));
  if len(I2) > 0:
    r = rrr[I2]; w[I2] = (1.0/8.0)*(3.0 + 2.0*r + np.sqrt(1.0 - 4.0*r - 4.0*r*r));
  if len(I3) > 0:
    r = rrr[I3]; w[I3] = (1.0/8.0)*(3.0 - 2.0*r + np.sqrt(1.0 + 4.0*r - 4.0*r*r));
  if len(I4) > 0:
    r = rrr[I4]; w[I4] = (1.0/8.0)*(5.0 - 2.0*r - np.sqrt(-7.0 + 12.0*r - 4.0*r*r));
  # w[I5] = 0.0;

  return w; 

def compute_matrix_Gamma_op(Y,params,extras=None):
  """ Compute the velocity averaging operator 
      $V_f = \Gamma[\mb{u}]$ to obtain 
      a reference particle velocity from the 
      surrounding fluid environment. 
  """
  # get params data 
  num_mesh_x = params['num_mesh_x'];
  num_mesh_y = params['num_mesh_y'];
  num_particles = params['num_particles'];
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaX = params['deltaX']; deltaV = deltaX_sq = deltaX*deltaX;

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_p,particle_theta = tuple(map(
    dd.get,['particle_q','particle_p','particle_theta']));
  fluid_phi,fluid_p,fluid_theta = tuple(map(
    dd.get,['fluid_phi','fluid_p','fluid_theta']));
  interface_q,interface_p,interface_theta = tuple(map(
    dd.get,['interface_q','interface_p','interface_theta']));

  if num_particles > 1: 
    ss = "Expecting only one particle for now, num_particles = " + str(num_particles);
    raise Exception(ss); 

  if num_dim != 2:
    raise Exception("assumes num_dim = 2, input gave num_dim = " + str(num_dim));

  if extras is not None:
    flag_save, = tuple(map(extras.get,['flag_save']));
  else:
    flag_save = None; 

  # default values
  if flag_save is None:
    flag_save = False; 

  # div: maps R^{dxd} -> R^d, (maps fluid velocity to particle velocity)
  matrix_Gamma_op = np.zeros((num_particles*num_dim,num_mesh_x*num_mesh_y*num_dim));

  # perform calculation of the velocity averaging operation 
  # in vicinity of the particle
  # assumes mesh is 0,L in each direction, where L = num_mesh_x*deltaX.
  L1 = num_mesh_x*deltaX; L2 = num_mesh_y*deltaX; 
  X0 = particle_q; # assumes just one particle 
  X1 = 0*particle_q;
  I0 = np.zeros(num_dim,dtype=int);
  II = np.zeros(num_dim,dtype=int);
  I0[:] = np.floor(X0/deltaX);
  I0[0] = np.mod(I0[0],num_mesh_x); I0[1] = np.mod(I0[1],num_mesh_y);
  if I0[0] < 0:
    I0[0] = num_mesh_x + I0[0];
  if I0[1] < 0:
    I0[1] = num_mesh_y + I0[1];
  # @optimize
  num_lattice = 5; one_over_a_sq = 1.0/(deltaX*deltaX);
  w = np.zeros((num_lattice,num_lattice)); 
  for i1 in range(0,num_lattice):
    for i2 in range(0,num_lattice):
      II[0] = I0[0] + i1 - 2; II[1] = I0[1] + i2 - 2; 
      X1 = II*deltaX; vec_R = X1 - X0; a = deltaX;
      f1 = peskin_delta(vec_R[0]/a); f2 = peskin_delta(vec_R[1]/a);
      w[i1,i2] = one_over_a_sq*f1[0]*f2[0];
       
      II1 = II[0]; 
      if II1 >= num_mesh_x:
        II1 = II1 - num_mesh_x;
      if II1 < 0:
        II1 = num_mesh_x + II1; 

      II2 = II[1]; 
      if II2 >= num_mesh_y:
        II2 = II2 - num_mesh_y;
      if II2 < 0:
        II2 = num_mesh_y + II2; 

      III = II2*num_mesh_x*num_dim + II1*num_dim; 
      matrix_Gamma_op[0,III] += w[i1,i2]*deltaX_sq; # integral operator (x-component)
      matrix_Gamma_op[1,III + 1] += w[i1,i2]*deltaX_sq; # integral operator (y-component)

  return matrix_Gamma_op;

def compute_matrix_Lambda_op(Y,params,extras=None):
  """ Compute the force spreading operator 
      $\mb{f} = \Lambda[\mb{F}]$ to obtain 
      a force density of the particle acting on 
      the surrounding fluid environment. 
  """
  if extras is None:
    matrix_Gamma_op = None;
  else:
    matrix_Gamma_op, = tuple(map(extras.get,['matrix_Gamma_op']));

  if matrix_Gamma_op is None:
     matrix_Gamma_op = compute_matrix_Gamma_op(Y,params,extras);

  matrix_Lambda_op = np.transpose(matrix_Gamma_op); # transpose 

  return matrix_Lambda_op;

def compute_M_S_j(Y,params):
  pass;

def compute_K0_j(Y,params):
  pass;

def compute_L0(Y,params):
  pass; 

def compute_K_ovdc_grad(Y,params,extras=None):

  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX \
    = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  bar_kappa,c0_conc \
    = tuple(map(params.get,['bar_kappa','c0_conc']));
  num_mesh_pts = num_mesh_x*num_mesh_y;
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;
  Y_I = params['Y_I'];

  qr = conc_q = get_comp(Y,'I1_conc_q','I2_conc_q',Y_I);

  if extras is not None:
    G, = tuple(map(extras.get,['G']));
  else:
    G = (None);

  if G is None:
    G = compute_matrix_vec_grad(Y,params);

  # K_ovdc case for concentration
  GG = G.reshape(num_mesh_pts,num_dim,num_mesh_pts);
  c0 = c0_conc;
  factor = np.expand_dims(qr*bar_kappa/(c0*deltaV),(1,2)); # deltaV since D_S_j has deltaV
  KK = factor*GG;
  K_ovdc_grad = KK.reshape(num_mesh_pts*num_dim,num_mesh_pts);

  return K_ovdc_grad;

def compute_dot_G(Y,params,extras=None):
 
  if extras is not None:
    G, = tuple(map(extras.get,['G']));
  else:
    G = None;

  if G is None:
    G = compute_matrix_vec_grad(Y,params);
 
  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX \
    = tuple(map(params.get,
      ['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  func_phi,extras_func_phi \
    = tuple(map(params.get,['func_phi','extras_func_phi']));

  Y_I,c0_conc = tuple(map(params.get,['Y_I','c0_conc']));

  conc_q = get_comp(Y,'I1_conc_q','I2_conc_q',Y_I);
  conc_theta = get_comp(Y,'I1_conc_theta','I2_conc_theta',Y_I);
  num_conc_theta = conc_theta.shape[0];

  phi,grad_r_phi,grad_X_phi = func_phi(Y,extras_func_phi);
  phi_vec = np.expand_dims(phi,1);
  c0 = c0_conc;
  dot_G = np.matmul(G,c0*phi_vec); # c0*grad Phi(x), we use the finite difference gradient here

  return dot_G;

def compute_K_ovdc_dot_G(Y,params,extras=None):

  if extras is not None:
    dot_G,G = tuple(map(extras.get,['dot_G','G']));
  else:
    dot_G,G = (None,None);

  if dot_G is None:
    dot_G = compute_dot_G(Y,params,{'G':G});

  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,
    ['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;

  bar_kappa,Y_I,c_v,c_v_I,c0_conc = tuple(map(params.get,
    ['bar_kappa','Y_I','c_v','c_v_I','c0_conc']));

  qr = conc_q = get_comp(Y,'I1_conc_q','I2_conc_q',Y_I);
  conc_theta = get_comp(Y,'I1_conc_theta','I2_conc_theta',Y_I);
  
  # local transpose the individual blocks of the tensor
  dot_GG = dot_G.reshape(num_mesh_x*num_mesh_y,num_dim);  
  c0 = c0_conc;
  factor = np.expand_dims(qr*bar_kappa/(c0*deltaV),1); # @@@ check expression (deltaV?)
  K_dFFF = factor*dot_GG; # density so scaling by 1/deltaV
  K_ovdc_dot_G = K_dFFF;

  return K_ovdc_dot_G; 

def compute_conc_K_heat2(Y,params,extras=None):
    
  # get params data
  kappa_C_C,num_mesh_x,num_mesh_y,num_dim,c_v,c_v_I,deltaX,Y_I = tuple(map(params.get,
    ['kappa_C_C','num_mesh_x','num_mesh_y','num_dim','c_v','c_v_I','deltaX','Y_I']));
  num_mesh_pts = num_mesh_x*num_mesh_y;
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;

  ii = c_v_I; 
  c_C = c_v[ii['conc']];

  conc_theta = get_comp(Y,'I1_conc_theta','I2_conc_theta',Y_I);
  num_conc_theta = conc_theta.shape[0];

  K_heat = np.zeros((num_conc_theta,num_conc_theta));
  x1 = np.linspace(0,num_mesh_x-1,num_mesh_x); 
  x2 = np.linspace(0,num_mesh_y-1,num_mesh_y);
  I = np.meshgrid(x1,x2); 
  I1 = np.rint(I[0].flatten()).astype(dtype=int);  
  I2 = np.rint(I[1].flatten()).astype(dtype=int);  
  II = np.array([I1,I2]).T; 

  Iip1 = np.zeros(II.shape,dtype=int); Iip1[:,:] = II; Iip1[:,0] = II[:,0] + 1; 
  Iim1 = np.zeros(II.shape,dtype=int); Iim1[:,:] = II; Iim1[:,0] = II[:,0] - 1;
  kk = np.nonzero(Iim1[:,0] < 0); Iim1[kk,0] = num_mesh_x + Iim1[kk,0];
  kk = np.nonzero(Iip1[:,0] >= num_mesh_x); Iip1[kk,0] = Iip1[kk,0] - num_mesh_x;
   
  Ijp1 = np.zeros(II.shape,dtype=int); Ijp1[:,:] = II; Ijp1[:,1] = II[:,1] + 1; 
  Ijm1 = np.zeros(II.shape,dtype=int); Ijm1[:,:] = II; Ijm1[:,1] = II[:,1] - 1;
  kk = np.nonzero(Ijm1[:,1] < 0); Ijm1[kk,1] = num_mesh_y + Ijm1[kk,1];
  kk = np.nonzero(Ijp1[:,1] >= num_mesh_y); Ijp1[kk,1] = Ijp1[kk,1] - num_mesh_y;
 
  II0 = II[:,1]*num_mesh_x + II[:,0];
  IIip1 = Iip1[:,1]*num_mesh_x + Iip1[:,0];
  IIim1 = Iim1[:,1]*num_mesh_x + Iim1[:,0];
  IIjp1 = Ijp1[:,1]*num_mesh_y + Ijp1[:,0];
  IIjm1 = Ijm1[:,1]*num_mesh_y + Ijm1[:,0];
  
  theta_I0 = conc_theta[II0];
  theta_Iip1 = conc_theta[IIip1];
  theta_Iim1 = conc_theta[IIim1];
  theta_Ijp1 = conc_theta[IIjp1];
  theta_Ijm1 = conc_theta[IIjm1];

  c = kappa_C_C/(c_C*c_C*deltaV);
  K_heat[II0,II0] = c*theta_I0*(theta_Iip1 + theta_Iim1 + theta_Ijp1 + theta_Ijm1);
  K_heat[II0,IIip1] = -c*theta_I0*theta_Iip1;
  K_heat[II0,IIim1] = -c*theta_I0*theta_Iim1;
  K_heat[II0,IIjp1] = -c*theta_I0*theta_Ijp1;
  K_heat[II0,IIjm1] = -c*theta_I0*theta_Ijm1;

  return K_heat; 

def compute_bar_K__conc1(Y,params,extras):

  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));

  get_parts = params['func_get_parts']; # function 

  num_mesh_x,num_mesh_y,num_dim,deltaX = \
    tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX; 

  Y_parts = dd = get_parts(Y,params);

  conc_q,conc_theta = tuple(map(
    dd.get,['conc_q','conc_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_conc_q,I2_conc_q = tuple(map(
    dd.get,['I1_conc_q','I2_conc_q']));
  I1_conc_theta,I2_conc_theta = tuple(map(
    dd.get,['I1_conc_theta','I2_conc_theta']));

  num_conc_q = conc_q.shape[0]; 
  num_conc_theta = conc_theta.shape[0]; 

  # extras handling   
  if extras is not None:
    extras_matrix_vec_div,extras_K_ovdc_grad = tuple(map(
      extras.get,['extras_matrix_vec_div','extras_K_ovdc_grad'])); 
    extras_K_ovdc_dot_G,extras_K_heat = tuple(map(
      extras.get,['extras_K_ovdc_dot_G','extras_K_heat']));
    flag_save,flag_save_energy_flux,flag_flux_check = tuple(map(
      extras.get,['flag_save','flag_save_energy_flux','flag_flux_check']));
  else:
    extras_matrix_vec_div,extras_K_ovdc_grad = (None,None);
    extras_K_ovdc_dot_G,extras_K_heat = (None,None);
    flag_save = None;  
    flag_save_energy_flux = None; 
    flag_flux_check = None;

  if flag_save is None:
    flag_save = False; 

  if flag_save_energy_flux is None:
    flag_save_energy_flux = False;
  
  if flag_flux_check is None:
    flag_flux_check = False;

  bar_K_j_list = [];
  I_in_list = []; I_out_list = []; 
  I_local_in_list = []; I_local_out_list = []; 
  if flag_save_energy_flux:
    energy_flux_list = [];  # one for each irreversible process 

  # concentration case
  j = c_v_I['conc'];  # j = 4  
  I1_q = 0; I2_q = I1_q + num_conc_q; 
  I1_theta = I2_q; I2_theta = I2_q + num_conc_theta;
  I_in = {'i1':[I1_conc_q,I1_conc_theta],
          'i2':[I2_conc_q,I2_conc_theta]};
  I_local_in = {'i1':[I1_q,I1_theta],'i2':[I2_q,I2_theta]};
  I_local_out = {'i1':[I1_q,I1_theta],'i2':[I2_q,I2_theta]};
  I_out = {'i1':[I1_conc_q,I1_conc_theta],
           'i2':[I2_conc_q,I2_conc_theta]};
  num_q = num_conc_q; 
  bar_K_j = np.zeros((num_q + num_conc_theta, num_q + num_conc_theta)); 
  
  ii = c_v_I;
  c_C = partial_theta_j_u_j = c_v[ii['conc']]; 
  theta_j = theta_C = conc_theta;
  matrix_vec_div = compute_matrix_vec_div(Y,params,extras_matrix_vec_div); 
  K_ovdc_grad = compute_K_ovdc_grad(Y,params,extras_K_ovdc_grad);
  dot_G = compute_dot_G(Y,params);
  flag_debug = False;
  if flag_debug:
    debug_dir = params['debug_dir'];
    xx = get_mesh_xx(params);
    aa = dot_G.reshape(num_mesh_pts,num_dim_sq)
    filename = debug_dir + '/dot_G_01.vtr';
    fff=[];
    for d in range(0,num_dim_sq):
      ff = {};
      ff['field_name'] = 'dot_G_%.2d'%d;
      ff['NumberOfComponents'] = 1;
      ff['field_values'] = aa[:,d]
      fff.append(ff);
    write_vtr_data(filename,xx,fff); 

  if extras_K_ovdc_dot_G is not None:
    extras_K_ovdc_dot_G.update({'dot_G':dot_G});
  K_ovdc_dot_G = compute_K_ovdc_dot_G(Y,params,extras_K_ovdc_dot_G);
  K_heat = compute_conc_K_heat2(Y,params,extras_K_heat);

  # --- put together the tensor K
  # compute the divergence of the gradient to obtain the upper-left block
  aa = -np.matmul(matrix_vec_div,K_ovdc_grad); # -\nabla \cdot K_ovdc \nabla 
  bar_K_j[I1_q:I2_q,I1_q:I2_q] = aa; 
  # Organize so each divergence tracked as if done separately for each
  # x-location after we already multipled in the c_C*dx/theta_C(x) term.
  aa = K_ovdc_dot_G; 
  K_ovdc_square_dot_G = np.zeros((num_mesh_pts*num_dim,num_mesh_pts)); # num_mesh_pts = num_theta 
  for d in range(0,num_dim):
    ii1 = range(d,num_mesh_pts*num_dim,num_dim); jj1 = range(0,num_mesh_pts); 
    K_ovdc_square_dot_G[ii1,jj1] = aa[:,d]; # setting subset of entries

  du = partial_theta_j_u_j;
  div_K_ovdc_square_dot_G_over_du = np.matmul(matrix_vec_div,K_ovdc_square_dot_G/du);
  bar_K_j[I1_q:I2_q,I1_theta:I2_theta] = div_K_ovdc_square_dot_G_over_du;

  # tranpose to get the other components of the tensor 
  bar_K_j[I1_theta:I2_theta,I1_q:I2_q] = np.transpose(bar_K_j[I1_q:I2_q,I1_theta:I2_theta]);
  # compute the lower-right block (dissipative work done that accumulates as internal energy tracked by temperature)
  # need to compute as if tensor dot product done at each x-location 
  ii2 = range(I1_theta,I2_theta); jj2 = range(I1_theta,I2_theta);
  for d in range(0,num_dim):
    bar_K_j[ii2,jj2] += dot_G[d::num_dim,0]*K_ovdc_dot_G[:,d]; # diagonal takes the square into account
  du = partial_theta_j_u_j;
  bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta] = bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta]/(du*du); # @optimize, only needs diagonal
  bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta] += K_heat/(du*du);

  bar_K_j_list.append(bar_K_j);
  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  # -- compute additional data
  if flag_save_energy_flux or flag_flux_check:
    rate_kinetic_conc = 0;
    rate_heat_conc = np.sum(np.dot(bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta],(c_C*deltaV/theta_C)))*c_C*deltaV;
    rate_total = rate_kinetic_conc + rate_heat_conc; 

  if flag_save_energy_flux:
    energy_flux = {};
    energy_flux['rate_kinetic_conc'] = rate_kinetic_conc; 
    energy_flux['rate_heat_conc'] = rate_heat_conc;
    energy_flux['rate_total'] = rate_total;
    energy_flux_list.append(energy_flux); 

  # check the energy exchanges (kinetic to heat energy)
  if flag_flux_check:
    print("");
    print("checking energy exchanges for conc:"); 
    print("rate_kinetic_conc = %.3e"%rate_kinetic_conc);
    print("rate_heat_conc = %.3e"%rate_heat_conc); 
    print("rate_total = %.1e"%rate_total);
    print("");
  
  # package the results to return  
  bar_K_j_indices = {'I_in':I_in_list,'I_out':I_out_list,
                     'I_local_in':I_local_in_list,'I_local_out':I_local_out_list};

  if flag_save_energy_flux:
    extras.update({'energy_flux_list':energy_flux_list}); 

  return bar_K_j_list, bar_K_j_indices; 

# particle (overdamped case)
def compute_D_j_inv_particle(Y,params,extras=None):
  gamma_particle,Y_I,c_v,c_v_I = tuple(map(params.get,['gamma_particle','Y_I','c_v','c_v_I']));
  num_particles,num_dim = tuple(map(params.get,['num_particles','num_dim']));

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  interface_q,interface_p,interface_theta = tuple(map(dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  num_q = num_particle_q;
  D_j_inv = (1.0/gamma_particle)*np.eye(num_q);  

  return D_j_inv;

def compute_K_ovd_particle(Y,params,extras=None):
  Y_I,c_v,c_v_I = tuple(map(params.get,['Y_I','c_v','c_v_I']));
  num_particles,num_dim = tuple(map(params.get,['num_particles','num_dim']));

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];

  D_j_inv = compute_D_j_inv_particle(Y,params,extras); 
  K_ovd = particle_theta*D_j_inv;

  return K_ovd;

def compute_partial_q_U(Y,params,extras=None):

  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));

  i1_q,i2_q = I1_particle_q,I2_particle_q;

  extras_D_E = {'flag_compute_parts':{'conc':False,'particle':True,
                                      'interface':False}};
  compute_D_E = params['func_compute_D_E'];
  D_E = compute_D_E(Y,params,extras_D_E); 
  partial_q_U = D_E[i1_q:i2_q];

  return partial_q_U; 

def compute_bar_K__overdamped_particle(Y,params,extras):

  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));

  get_parts = params['func_get_parts']; # function 

  num_mesh_x,num_mesh_y,num_dim,deltaX = \
    tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX; 

  Y_parts = dd = get_parts(Y,params);

  particle_q,particle_theta = tuple(map(
    dd.get,['particle_q','particle_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(
    dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; 
  num_particle_theta = particle_theta.shape[0]; 

  # extras handling   
  if extras is not None:
    extras_matrix_vec_div,extras_K_ovdc_grad = tuple(map(
      extras.get,['extras_matrix_vec_div','extras_K_ovdc_grad'])); 
    extras_K_ovdc_dot_G,extras_K_heat = tuple(map(
      extras.get,['extras_K_ovdc_dot_G','extras_K_heat']));
    flag_save,flag_save_energy_flux,flag_flux_check = tuple(map(
      extras.get,['flag_save','flag_save_energy_flux','flag_flux_check']));
  else:
    extras_matrix_vec_div,extras_K_ovdc_grad = (None,None);
    extras_K_ovdc_dot_G,extras_K_heat = (None,None);
    flag_save = None;  
    flag_save_energy_flux = None; 
    flag_flux_check = None;

  if flag_save is None:
    flag_save = False; 

  if flag_save_energy_flux is None:
    flag_save_energy_flux = False;
  
  if flag_flux_check is None:
    flag_flux_check = False;

  bar_K_j_list = [];
  I_in_list = []; I_out_list = []; 
  I_local_in_list = []; I_local_out_list = []; 
  if flag_save_energy_flux:
    energy_flux_list = [];  # one for each irreversible process 

  # particle case
  j = c_v_I['particle'];  # j = 4  
  I1_q = 0; I2_q = I1_q + num_particle_q; 
  I1_theta = I2_q; I2_theta = I2_q + num_particle_theta;
  I_in = {'i1':[I1_particle_q,I1_particle_theta],
          'i2':[I2_particle_q,I2_particle_theta]};
  I_local_in = {'i1':[I1_q,I1_theta],'i2':[I2_q,I2_theta]};
  I_local_out = {'i1':[I1_q,I1_theta],'i2':[I2_q,I2_theta]};
  I_out = {'i1':[I1_particle_q,I1_particle_theta],
           'i2':[I2_particle_q,I2_particle_theta]};
  num_q = num_particle_q; 
  bar_K_j = np.zeros((num_q + num_particle_theta, num_q + num_particle_theta)); 

  
  ii = c_v_I;
  c_P = partial_theta_j_u_j = c_v[ii['particle']]; 
  theta_j = theta_P = particle_theta;
  theta_P__D_inv = K_ovd_particle = compute_K_ovd_particle(Y,params,extras);
  dq_U = partial_q_U = compute_partial_q_U(Y,params); # compute energy change in q
  flag_debug = False;
  if flag_debug:
    debug_dir = params['debug_dir'];
    xx = get_mesh_xx(params);
    aa = dot_G.reshape(num_mesh_pts,num_dim_sq)
    filename = debug_dir + '/dot_G_01.vtr';
    fff=[];
    for d in range(0,num_dim_sq):
      ff = {};
      ff['field_name'] = 'dot_G_%.2d'%d;
      ff['NumberOfComponents'] = 1;
      ff['field_values'] = aa[:,d]
      fff.append(ff);
    write_vtr_data(filename,xx,fff); 

  # --- put together the tensor K
  # compute the divergence of the gradient to obtain the upper-left block
  bar_K_j[I1_q:I2_q,I1_q:I2_q] = K_ovd_particle; 
  # Organize so done separately for each x-location
  theta_P__D_inv__dq_U = np.matmul(theta_P__D_inv,dq_U); 
  bb = np.expand_dims(theta_P__D_inv__dq_U,1);
  bar_K_j[I1_q:I2_q,I1_theta:I2_theta] = -1.0*bb/c_P;
  # tranpose to get the other components of the tensor 
  bar_K_j[I1_theta:I2_theta,I1_q:I2_q] = np.transpose(bar_K_j[I1_q:I2_q,I1_theta:I2_theta]);

  # compute the lower-right block (dissipative work done that accumulates as internal energy tracked by temperature)
  # need to compute as if tensor dot product done at each x-location 
  cc = dq_U*theta_P__D_inv__dq_U; 
  ii2 = range(I1_theta,I2_theta); jj2 = range(I1_theta,I2_theta);
  bar_K_j[ii2,jj2] += 1.0*np.sum(cc)/(c_P*c_P); # diagonal takes the square into account

  bar_K_j_list.append(bar_K_j);
  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  # -- compute additional data
  if flag_save_energy_flux or flag_flux_check:
    rate_kinetic_particle = 0;
    rate_heat_particle = np.sum(np.dot(bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta],(c_P/theta_P)))*c_P;
    rate_total = rate_kinetic_particle + rate_heat_particle; 

  if flag_save_energy_flux:
    energy_flux = {};
    energy_flux['rate_kinetic_particle'] = rate_kinetic_particle; 
    energy_flux['rate_heat_particle'] = rate_heat_particle;
    energy_flux['rate_total'] = rate_total;
    energy_flux_list.append(energy_flux); 

  # check the energy exchanges (kinetic to heat energy)
  if flag_flux_check:
    print("");
    print("checking energy exchanges for particle:"); 
    print("rate_kinetic_particle = %.3e"%rate_kinetic_particle);
    print("rate_heat_particle = %.3e"%rate_heat_particle); 
    print("rate_total = %.1e"%rate_total);
    print("");
  
  # package the results to return  
  bar_K_j_indices = {'I_in':I_in_list,'I_out':I_out_list,
                     'I_local_in':I_local_in_list,'I_local_out':I_local_out_list};

  if flag_save_energy_flux:
    extras.update({'energy_flux_list':energy_flux_list}); 

  return bar_K_j_list, bar_K_j_indices; 

def compute_D_j_interface_conc(Y,params,extras=None):
  Y_I,c_v,c_v_I,deltaX,gamma = tuple(map(
    params.get,['Y_I','c_v','c_v_I','deltaX','gamma']));
  num_particles,num_dim = tuple(map(params.get,['num_particles','num_dim']));
  deltaV = deltaX_sq = deltaX*deltaX; 

  if extras is not None:
    flag_save, = tuple(map(extras.get,['flag_save']));
  else:
    flag_save = None; 

  if flag_save is None:
    flag_save = False; 

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_conc_q,I2_conc_q = tuple(map(dd.get,['I1_conc_q','I2_conc_q']));
  I1_conc_theta,I2_conc_theta = tuple(map(dd.get,['I1_conc_theta','I2_conc_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  num_q = num_particle_q + num_conc_q;
  D_j = np.zeros((num_q, num_q));  # assumes only one temperature to track accumulated dissipation energy
 
  # particle drag term from the interface 
  matrix_Gamma_op = compute_matrix_Gamma_op(Y,params); 

  if flag_save:
    extras.update({'matrix_Gamma_op':matrix_Gamma_op});

  return D_j;

def compute_conc_particle_interface(Y,params,extras): 

  bar_K_j_list = [];
  I_in_list = []; I_out_list = []; 
  I_local_in_list = []; I_local_out_list = []; 

  Y_I,c_v,c_v_I = tuple(map(params.get,['Y_I','c_v','c_v_I']));
  num_particles,num_dim = tuple(map(params.get,['num_particles','num_dim']));

  kappa_C_I,kappa_P_I = tuple(map(params.get,['kappa_C_I','kappa_P_I']));

  num_mesh_x,num_mesh_y,num_dim,deltaX = \
    tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX; 

  get_parts = params['func_get_parts'];

  # extras handling   
  if extras is not None:
    extras_matrix_vec_div,extras_K_ovdc_grad = tuple(map(
      extras.get,['extras_matrix_vec_div','extras_K_ovdc_grad'])); 
    extras_K_ovdc_dot_G,extras_K_heat = tuple(map(
      extras.get,['extras_K_ovdc_dot_G','extras_K_heat']));
    flag_save,flag_save_energy_flux,flag_flux_check = tuple(map(
      extras.get,['flag_save','flag_save_energy_flux','flag_flux_check']));
  else:
    extras_matrix_vec_div,extras_K_ovdc_grad = (None,None);
    extras_K_ovdc_dot_G,extras_K_heat = (None,None);
    flag_save = None;  
    flag_save_energy_flux = None; 
    flag_flux_check = None;

  if flag_save is None:
    flag_save = False; 

  if flag_save_energy_flux is None:
    flag_save_energy_flux = False;
  
  if flag_flux_check is None:
    flag_flux_check = False;

  if flag_save_energy_flux:
    energy_flux_list = [];  

  Y_parts = dd = get_parts(Y,params);
  conc_q, conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  particle_q, particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_conc_q,I2_conc_q = tuple(map(dd.get,['I1_conc_q','I2_conc_q']));
  I1_conc_theta,I2_conc_theta = tuple(map(dd.get,['I1_conc_theta','I2_conc_theta']));
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));
  I1_interface_q,I2_interface_q = tuple(map(dd.get,['I1_interface_q','I2_interface_q']));
  I1_interface_theta,I2_interface_theta = tuple(map(dd.get,['I1_interface_theta','I2_interface_theta']));

  num_conc_q = conc_q.shape[0]; num_conc_theta = conc_theta.shape[0];
  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  num_q = num_particle_q;

  extras_D_j_interface = {'flag_save':True};
  D_j = compute_D_j_interface_conc(Y,params,extras_D_j_interface); ii = c_v_I;

  # interface case
  j = c_v_I['interface'];  # j = 2
  if num_interface_theta > 1:
    ss = ""; ss += "Expected num_interface_theta = 1, but num_interface_theta = " + str(num_interface_theta);
    ss += "Need a separate K_j for each interface, currently not implemented.";
    raise Exception(ss);
  num_q = num_particle_q + num_conc_q; num_theta = num_particle_theta + num_conc_theta + num_interface_theta;
  I1_q1 = 0; I2_q1 = I1_q1 + num_particle_q; 
  I1_q2 = I2_q1; I2_q2 = I1_q2 + num_conc_q; 
  I1_theta1 = I2_q2; I2_theta1 = I1_theta1 + num_particle_theta; # particle theta 
  I1_theta2 = I2_theta1; I2_theta2 = I1_theta2 + num_conc_theta; # conc theta 
  I1_theta3 = I2_theta2; I2_theta3 = I1_theta3 + num_interface_theta; # interface theta 

  I1_q = I1_q1; I2_q = I2_q2; 
  I1_theta = I1_theta1; I2_theta = I2_theta3;

  I_in = {'i1':[I1_particle_q,I1_conc_q,I1_particle_theta,I1_conc_theta,I1_interface_theta],
          'i2':[I2_particle_q,I2_conc_q,I2_particle_theta,I2_conc_theta,I2_interface_theta]};
  I_local_in = {'i1':[I1_q1,I1_q2,I1_theta1,I1_theta2,I1_theta3],
                'i2':[I2_q1,I2_q2,I2_theta1,I2_theta2,I2_theta3]};
  I_local_out = {'i1':[I1_q1,I1_q2,I1_theta1,I1_theta2,I1_theta3],
                 'i2':[I2_q1,I2_q2,I2_theta1,I2_theta2,I2_theta3]};
  I_out = {'i1':[I1_particle_q,I1_conc_q,I1_particle_theta,I1_conc_theta,I1_interface_theta],
           'i2':[I2_particle_q,I2_conc_q,I2_particle_theta,I2_conc_theta,I2_interface_theta]};

  bar_K_j = np.zeros((num_q + num_theta, num_q + num_theta)); 
 
  ii = c_v_I;
  d_tau_u = C_I = partial_theta_I_u_j = partial_theta_j_u_j = c_v[ii['interface']];

  # heat exchnage terms between the interface, conc, and particle(s)   
  ii = c_v_I;
  c_P = cc_P = c_v[ii['particle']]; c_C = cc_C = c_v[ii['conc']]; 
  c_I = cc_I = c_v[ii['interface']]; 
  c_I_inv_dx = cc_I_inv_dx = cc_I/deltaV; 
  c_C_inv_dx = cc_C_inv_dx = cc_C/deltaV; 
  c_I_I = partial_tau_u_I_I = cc_I*cc_I; c_I_I_inv_dx = cc_I*cc_I_inv_dx;
  c_C_C = partial_tau_u_C_C = cc_C*cc_C; c_C_C_inv_dx = cc_C*cc_C_inv_dx;
  c_P_P = partial_tau_u_P_P = cc_P*cc_P; c_P_I = partial_tau_u_P_I = cc_P*cc_I; 
  c_C_I = partial_tau_u_C_I = cc_C*cc_I; c_C_I_inv_dx = cc_C*cc_I_inv_dx;
  theta_P = particle_theta; theta_C = conc_theta; theta_I = interface_theta; 
  Gamma_op_vec = matrix_Gamma_op = extras_D_j_interface['matrix_Gamma_op'];
  aa = Gamma_op_vec.reshape(num_particles,num_dim,num_mesh_pts,num_dim);
  Gamma_op = Gamma_op_scalar = aa[:,0,:,0].reshape(num_particles,num_mesh_pts); # get scalar operator
  Lambda_op = Gamma_op.T;
  kappa_C_I_xx_dx = kappa_C_I*Lambda_op.flatten();  # spatial dependence of thermal conductivity
  kappa_C_I_xx = kappa_C_I_xx_dx/deltaV;  # spatial dependence of thermal conductivity (so integrates)
  
  bar_K_j[I1_theta1:I2_theta1,I1_theta1:I2_theta1] = kappa_P_I*theta_I*theta_P/c_P_P; 
  bar_K_j[I1_theta1:I2_theta1,I1_theta3:I2_theta3] = -kappa_P_I*theta_P*theta_I/c_P_I;
  ii1 = range(I1_theta2,I2_theta2); ii2 = range(I1_theta2,I2_theta2);
  bar_K_j[ii1,ii2] = kappa_C_I_xx*theta_C*theta_I/(c_C_C*deltaV); # diagonal  
  bar_K_j[I1_theta2:I2_theta2,I1_theta3:I2_theta3] = np.expand_dims(-kappa_C_I_xx*theta_I*theta_C/c_C_I,1);
  bar_K_j[I1_theta3:I2_theta3,I1_theta1:I2_theta1] = -kappa_P_I*theta_I*theta_P/c_P_I;
  bar_K_j[I1_theta3:I2_theta3,I1_theta2:I2_theta2] = bar_K_j[I1_theta2:I2_theta2,I1_theta3:I2_theta3].transpose();
  bar_K_j[I1_theta3:I2_theta3,I1_theta3:I2_theta3] \
    += ((kappa_P_I*theta_P*theta_I)/c_I_I) + (np.sum(kappa_C_I_xx*theta_I*theta_C*deltaV)/c_I_I); # summing energy density, hence deltaV.

  # save the operator 
  bar_K_j_list.append(bar_K_j);
  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  # package the results to return  
  bar_K_j_indices = {'I_in':I_in_list,'I_out':I_out_list,
                     'I_local_in':I_local_in_list,'I_local_out':I_local_out_list};

  # -- compute additional data
  if flag_save_energy_flux or flag_flux_check:
    rate_kinetic_particle = 0;
    rate_heat_particle = 0.0; 
    rate_total = rate_kinetic_particle + rate_heat_particle; 

  if flag_save_energy_flux:
    energy_flux = {};
    energy_flux['rate_kinetic_particle'] = rate_kinetic_particle; 
    energy_flux['rate_heat_particle'] = rate_heat_particle;
    energy_flux['rate_total'] = rate_total;
    energy_flux_list.append(energy_flux); 

  if flag_save_energy_flux:
    extras.update({'energy_flux_list':energy_flux_list}); 

  return bar_K_j_list, bar_K_j_indices; 

def compute_bar_L_conc(Y,params,extras=None):
  Y_I,c_v,c_v_I,deltaX = tuple(map(
    params.get,['Y_I','c_v','c_v_I','deltaX']));
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX;

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(
    dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(
    dd.get,['conc_q','conc_theta']));
  interface_q,interface_p,interface_theta = tuple(map(
    dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(
    dd.get,['I1_particle_theta','I2_particle_theta']));

  I1_conc_q,I2_conc_q = tuple(map(
    dd.get,['I1_conc_q','I2_conc_q']));
  I1_conc_theta,I2_conc_theta = tuple(map(
    dd.get,['I1_conc_theta','I2_conc_theta']));

  I1_interface_q,I2_interface_q = tuple(map(    
    dd.get,['I1_interface_q','I2_interface_q']));
  I1_interface_theta,I2_interface_theta = tuple(map(
    dd.get,['I1_interface_theta','I2_interface_theta']));

  num_particle_q = particle_q.shape[0];
  num_particle_theta = particle_theta.shape[0]; 
  num_conc_q = conc_q.shape[0];
  num_conc_theta = conc_theta.shape[0]; 
  num_interface_q = interface_q.shape[0]; 
  num_interface_theta = interface_theta.shape[0];

  # extras handling   
  if extras is not None:
    D_S_j,flag_save = tuple(map(
      extras.get,['D_S_j','flag_save'])); 
  else:
    D_S_j,flag_save = (None,None);

  if flag_save is None:
    flag_save = False; 

  if D_S_j is None:
    D_S_j = compute_D_S_j(Y_n,params);

  # compute the bar_L operator 
  num_Y = Y.shape[0];
  bar_L = np.zeros((num_Y, num_Y)); 

  # there is a separate entropy $S^{(j)}$ for each heat-body 
  I_in = {'i1':[0],'i2':[Y.shape[0]]};
  I_local_in = {'i1':[0],'i2':[Y.shape[0]]};
  I_local_out = {'i1':[0],'i2':[Y.shape[0]]};
  I_out = {'i1':[0],'i2':[Y.shape[0]]};

  bar_L_indices = {'I_in':I_in,'I_out':I_out,
                   'I_local_in':I_local_in,'I_local_out':I_local_out};

  return bar_L,bar_L_indices;  

def compute_bar_K_j_conc(Y,params,extras): 

  flag_save_energy_flux,energy_flux_list = \
    tuple(map(extras.get,['flag_save_energy_flux',
                          'energy_flux_list']));
    
  flag_save,flag_flux_check = \
    tuple(map(extras.get,['flag_save',
                          'flag_flux_check']));

  if flag_save_energy_flux:
    if energy_flux_list is None:
      extras['energy_flux_list'] = energy_flux_list = []; # create list

  if flag_save is None:
    flag_save = True; 

  if flag_flux_check is None:
    flag_flux_check = True; 

  extras_bar_K_list = [];

  extras_bar_K_conc = {'flag_save':flag_save,
                       'flag_save_energy_flux':flag_save_energy_flux,
                       'flag_flux_check':flag_flux_check}
  bar_K_conc, bar_K_conc_indices = compute_bar_K__conc1(Y,params,extras_bar_K_conc); 
  extras_bar_K_list.append(extras_bar_K_conc);

  extras_bar_K_particle = {'flag_save':flag_save,
                           'flag_save_energy_flux':flag_save_energy_flux,
                           'flag_flux_check':flag_flux_check};
  bar_K_particle, bar_K_particle_indices = \
    compute_bar_K__overdamped_particle(Y,params,extras_bar_K_particle);
  extras_bar_K_list.append(extras_bar_K_particle);

  extras_bar_K_interface = {'flag_save':flag_save,
                            'flag_save_energy_flux':flag_save_energy_flux,
                            'flag_flux_check':flag_flux_check};
  bar_K_interface, bar_K_interface_indices = \
    compute_conc_particle_interface(Y,params,extras_bar_K_interface);
  extras_bar_K_list.append(extras_bar_K_interface);

  # collect info for returning results
  bar_K_j_list = bar_K_particle + bar_K_conc + bar_K_interface; 
  bar_K_j_indices_list_raw = [bar_K_particle_indices, bar_K_conc_indices,bar_K_interface_indices]; 
  # convert each key to an array
  bar_K_j_indices = {}; key_list = list(bar_K_j_indices_list_raw[0].keys());
  for k in key_list:
    bar_K_j_indices[k] = [];
  for ii in range(0,len(bar_K_j_indices_list_raw)):
    for k in key_list:
      bar_K_j_indices[k] += bar_K_j_indices_list_raw[ii][k]; # concat lists

  if flag_save_energy_flux:
    for ii in range(0,len(bar_K_j_list)):
      energy_flux_list += extras_bar_K_list[ii]['energy_flux_list'];
    extras.update({'energy_flux_list':energy_flux_list}); 

  return bar_K_j_list,bar_K_j_indices; 

def compute_R_ovd_particle(Y,params,extras=None):
  """ R_ovd factor so that R_ovd*R_ovd^T = K_ovdc
  used for computing thermal fluctuations for particles.
  """ 
  gamma_particle,Y_I = tuple(map(params.get,['gamma_particle','Y_I']));
  num_dim, = tuple(map(params.get,['num_dim']));

  particle_q = get_comp(Y,'I1_particle_q','I2_particle_q',Y_I);
  particle_theta = get_comp(Y,'I1_particle_theta','I2_particle_theta',Y_I);
  num_particle_q = particle_q.shape[0];
  num_particle_theta = particle_theta.shape[0];

  if extras is not None:
    flag_save, = \
      tuple(map(extras.get,['flag_save']));
  else:
    flag_save = None;
  
  if flag_save is None:
    flag_save = False;
 
  theta_P = particle_theta; 
  R_ovd = np.sqrt(theta_P/gamma_particle)*np.eye(num_particle_q);
    
  return R_ovd; 

def compute_R_ovdc(Y,params,extras=None):
  """ R_ovdc factor so that R_ovdc*R_ovdc^T = K_ovdc
  used for computing thermal fluctuations of concentration fields.
  """ 
  bar_kappa,c0_conc,Y_I = tuple(map(params.get,['bar_kappa','c0_conc','Y_I']));
  num_mesh_x,num_mesh_y,num_dim,deltaX = \
    tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX;
 
  if extras is not None:
    extras_matrix_vec_div,extras_K_ovdc_dot_G = \
      tuple(map(extras.get,['extras_matrix_vec_div','extras_K_ovdc_dot_G'])); 

    if extras_matrix_vec_div is None: 
      extras['extras_matrix_vec_div'] = {}; # for saving data

    if extras_K_ovdc_dot_G is None:
      extras['extras_K_ovdc_dot_G'] = {};
     
  else:
    extras_matrix_vec_div = None; 
    extras_K_ovdc_dot_G = None; 

  conc_theta = get_comp(Y,'I1_conc_theta','I2_conc_theta',Y_I);
  conc_q = get_comp(Y,'I1_conc_q','I2_conc_q',Y_I);

  if extras is not None:
    flag_save,matrix_vec_div,dot_G = \
      tuple(map(extras.get,['flag_save','matrix_vec_div','dot_G']));
  else:
    flag_save,matrix_vec_div,dot_G = (None,None,None);
  
  if flag_save is None:
    flag_save = False;

  if matrix_vec_div is None:
    matrix_vec_div = compute_matrix_vec_div(Y,params,extras_matrix_vec_div); 

  if dot_G is None:
    extras_K_ovdc_dot_G = extras['extras_K_ovdc_dot_G'];
    dot_G = compute_dot_G(Y,params,extras_K_ovdc_dot_G);
 
  R_ovdc = np.zeros((num_mesh_x*num_mesh_y*num_dim,
                     num_mesh_x*num_mesh_y*num_dim));
  c0 = c0_conc; 
  for i1 in range(0,num_mesh_x):
    for j1 in range(0,num_mesh_y):
      I0_mesh = j1*num_mesh_x + i1;
      I_qq = j1*num_mesh_x + i1;
      sqrt_bar_kappa_qr_c0 = np.sqrt(bar_kappa*conc_q[I_qq]/c0);
      for a in range(0,num_dim):
        R_ovdc[I0_mesh*num_dim + a,I0_mesh*num_dim + a] \
          += sqrt_bar_kappa_qr_c0;
  
  R_ovdc = R_ovdc/np.sqrt(deltaV); # scaling by (1/deltaV)^{1/2} needed for density
 
  if extras is not None and 'dot_G' in extras: 
    extras.update({'dot_G':None}); 

  if flag_save: 
    extras.update({'matrix_vec_div':matrix_vec_div}); 
    
  return R_ovdc; 

def compute_R_heat2__conc(Y,params,extras):
  """ Compute the factor $K_heat = RR^T$. """

  # get params data
  kappa_C_C,c0_conc,num_mesh_x,num_mesh_y,num_dim,c_v,c_v_I,deltaX,mu,Y_I = \
    tuple(map(params.get,['kappa_C_C','c0_conc','num_mesh_x','num_mesh_y',
                          'num_dim','c_v','c_v_I','deltaX','mu','Y_I']));

  num_mesh_pts = num_mesh_x*num_mesh_y;
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;

  ii = c_v_I; 
  c_C = c_v[ii['conc']];

  conc_theta = get_comp(Y,'I1_conc_theta','I2_conc_theta',Y_I);
  num_conc_theta = conc_theta.shape[0];

  wG = np.zeros((num_conc_theta,num_dim,num_conc_theta)); # weighted gradient
  x1 = np.linspace(0,num_mesh_x-1,num_mesh_x); 
  x2 = np.linspace(0,num_mesh_y-1,num_mesh_y);
  I = np.meshgrid(x1,x2); 
  I1 = np.rint(I[0].flatten()).astype(dtype=int);  
  I2 = np.rint(I[1].flatten()).astype(dtype=int);  
  II = np.array([I1,I2]).T; 

  Iip1 = np.zeros(II.shape,dtype=int); Iip1[:,:] = II; Iip1[:,0] = II[:,0] + 1; 
  Iim1 = np.zeros(II.shape,dtype=int); Iim1[:,:] = II; Iim1[:,0] = II[:,0] - 1;
  kk = np.nonzero(Iim1[:,0] < 0); Iim1[kk,0] = num_mesh_x + Iim1[kk,0];
  kk = np.nonzero(Iip1[:,0] >= num_mesh_x); Iip1[kk,0] = Iip1[kk,0] - num_mesh_x;
   
  Ijp1 = np.zeros(II.shape,dtype=int); Ijp1[:,:] = II; Ijp1[:,1] = II[:,1] + 1; 
  Ijm1 = np.zeros(II.shape,dtype=int); Ijm1[:,:] = II; Ijm1[:,1] = II[:,1] - 1;
  kk = np.nonzero(Ijm1[:,1] < 0); Ijm1[kk,1] = num_mesh_y + Ijm1[kk,1];
  kk = np.nonzero(Ijp1[:,1] >= num_mesh_y); Ijp1[kk,1] = Ijp1[kk,1] - num_mesh_y;
 
  II0 = II[:,1]*num_mesh_x + II[:,0];
  IIip1 = Iip1[:,1]*num_mesh_x + Iip1[:,0];
  IIim1 = Iim1[:,1]*num_mesh_x + Iim1[:,0];
  IIjp1 = Ijp1[:,1]*num_mesh_y + Ijp1[:,0];
  IIjm1 = Ijm1[:,1]*num_mesh_y + Ijm1[:,0];
  
  theta_I0 = conc_theta[II0];
  theta_Iip1 = conc_theta[IIip1];
  theta_Iim1 = conc_theta[IIim1];
  theta_Ijp1 = conc_theta[IIjp1];
  theta_Ijm1 = conc_theta[IIjm1];

  c = np.sqrt(kappa_C_C/(c_C*c_C*deltaV));
  wG[II0,0,II0] = c*np.sqrt(theta_I0*theta_Iim1);
  wG[II0,0,IIim1] = -c*np.sqrt(theta_I0*theta_Iim1);
  wG[II0,1,II0] = c*np.sqrt(theta_I0*theta_Ijm1);
  wG[II0,1,IIjm1] = -c*np.sqrt(theta_I0*theta_Ijm1);

  wwG = wG.reshape(II0.shape[0]*num_dim,II0.shape[0]);

  R_heat = wwG.T;
  
  return R_heat; 

def compute_B_j__particle(Y,params,extras): 

  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));
  num_mesh_x,num_mesh_y,deltaX = tuple(map(
    params.get,['num_mesh_x','num_mesh_y','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX; 
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  gamma_particle, = tuple(map(
    params.get,['gamma_particle']));
  num_dim = params['num_dim']; 
  num_dim_sq = num_dim*num_dim; 
  k_B = params['k_B'];
  sqrt_two_k_B = np.sqrt(2.0*k_B);

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(
    dd.get,['particle_q','particle_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(
   dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; 
  num_particle_theta = particle_theta.shape[0]; 

  if extras is not None: 
    flag_save, = tuple(map(extras.get,['flag_save']));
  else:
    flag_save = None;

  if flag_save is None:
    flag_save = False; 
  
  ii = c_v_I;
  j = ii['particle']; 
  partial_tau_U = C_P = c_v[j];

  n1 = num_particle_q + num_particle_theta;
  n2 = num_particle_q + num_particle_theta;
  B_j = np.zeros((n1,n2));

  # compute the entries
  R_ovd = compute_R_ovd_particle(Y,params,extras); 
  partial_q_U = compute_partial_q_U(Y,params); 
  partial_q_U_T = partial_q_U.T; 

  theta_P = particle_theta; 

  i1 = 0; i2 = i1 + num_particle_q;
  j1 = 0; j2 = j1 + num_particle_q;
  B_j[i1:i2,j1:j2] = R_ovd; #np.sqrt(theta_P)*R_D_inv; 
  i2_theta_start = i2;
  i1 = i2_theta_start; i2 = i1 + num_particle_theta;
  j1 = 0; j2 = j1 + num_particle_q;
  aa = np.matmul(-partial_q_U_T,R_ovd)/partial_tau_U; 
  B_j[i1:i2,j1:j2] = aa; 

  I_in = None; # not relevant for noise 
  I1_q = 0; I2_q = I1_q + num_particle_q;
  I1_theta = I2_q; I2_theta = I1_theta + num_particle_theta;
  I_local_in = None; # not relevant for noise 
  I_local_out = {'i1':[I1_q,I1_theta],'i2':[I2_q,I2_theta]};  
  I_out = {'i1':[I1_particle_q,I1_particle_theta],'i2':[I2_particle_q,I2_particle_theta]};

  B_j = sqrt_two_k_B*B_j; # scale so BB^T = 2*k_B*K 

  B_j_indexing = {'I_in':I_in,'I_out':I_out,
                  'I_local_in':I_local_in,'I_local_out':I_local_out};

  return B_j,B_j_indexing; 

def compute_B_j__conc_field(Y,params,extras): 

  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));
  num_mesh_x,num_mesh_y,deltaX = tuple(map(
    params.get,['num_mesh_x','num_mesh_y','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX; 
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  num_dim = params['num_dim']; 
  num_dim_sq = num_dim*num_dim; 
  k_B = params['k_B'];
  sqrt_two_k_B = np.sqrt(2.0*k_B);
  flag_incompressible = params['flag_incompressible'];

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  conc_q,conc_theta = tuple(map(
    dd.get,['conc_q','conc_theta']));

  Y_I_parts = dd = get_parts_I(params);

  I1_conc_q,I2_conc_q = tuple(map(
    dd.get,['I1_conc_q','I2_conc_q']));
  I1_conc_theta,I2_conc_theta = tuple(map(
    dd.get,['I1_conc_theta','I2_conc_theta']));

  num_conc_q = conc_q.shape[0];  
  num_conc_theta = conc_theta.shape[0]; 
  
  if extras is not None:
    bar_K_j,bar_K_j_indices,matrix_vec_div,flag_save = \
      tuple(map(extras.get,['bar_K_j','bar_K_j_indices',
            'matrix_vec_div','flag_save']));
    extras_matrix_vec_div,extras_K_ovdc_dot_G = \
      tuple(map(extras.get,['extras_matrix_vec_div',
                            'extras_K_ovdc_dot_G']));
    matrix_Lambda_op, = tuple(map(extras.get,
                        ['matrix_Lambda_op']));
  else:
    bar_K_j,bar_K_j_indices,matrix_vec_div,flag_save = \
      (None,None,None,None);
    extras_matrix_vec_div,extras_K_ovdc_dot_G = \
      (None,None);
    matrix_Lambda_op = None;

  if flag_save is None:
    flag_save = False; 
 
  extras_R_ovdc = {};
  if extras_matrix_vec_div is not None:
    extras_R_ovdc.update({'matrix_vec_div':extras_matrix_vec_div['D'],
                          'extras_K_ovdc_dot_G':extras_K_ovdc_dot_G});
    matrix_vec_div = extras_matrix_vec_div['D'];

  R_ovdc = compute_R_ovdc(Y,params,extras_R_ovdc);

  extras_R_heat = {};
  if extras_matrix_vec_div is not None:
    extras_R_heat.update({'matrix_vec_div':extras_matrix_vec_div['D']});
  if extras_K_ovdc_dot_G is not None:  # @@@ change for conc operators
    extras_R_heat.update({'dot_G':extras_K_ovdc_dot_G['dot_G']});
    dot_G = extras_K_ovdc_dot_G['dot_G'];
  else:
    dot_G = None; 

  R_heat = compute_R_heat2__conc(Y,params,extras_R_heat);

  if dot_G is None:
    dot_G = compute_dot_G(Y,params);

  if matrix_vec_div is None:
    matrix_vec_div = compute_matrix_vec_div(Y,params);

  if matrix_Lambda_op is None:
    matrix_Lambda_op = compute_matrix_Lambda_op(Y,params);

  ii = c_v_I; 
  j = ii['conc']; c_C = c_v[j]; 
  n1 = num_conc_q + num_conc_theta;
  n2 = num_conc_q*num_dim + num_conc_theta*num_dim;
  B_j = np.zeros((n1,n2));
  theta_C = conc_theta;

  # use the computed R_ovdc and R_heat to construct B_j
  i1 = 0; i2 = i1 + num_conc_q;
  j1 = 0; j2 = j1 + num_conc_q*num_dim;
  aa = -np.dot(matrix_vec_div,R_ovdc);
  B_j[i1:i2,j1:j2] = aa; 
  i2_theta_start = i2;

  i1 = i2_theta_start; i2 = i1 + num_conc_theta;
  j1 = 0; j2 = j1 + num_mesh_pts*num_dim;
  FF = dot_G.reshape(num_mesh_pts,num_dim,1,1); 
  A = R_ovdc/c_C; AA = A.reshape(num_mesh_pts,num_dim,num_mesh_pts,num_dim);
  CC = -np.sum(FF*AA,1); # contraction ->  -F:R_ovdc/c_C.
  C = CC.reshape(num_mesh_pts,num_mesh_pts*num_dim);
  B_j[i1:i2,j1:j2] += C;

  i1 = i2_theta_start; i2 = i1 + num_conc_theta;
  j1 = j2; j2 = j1 + num_conc_theta*num_dim;
  B_j[i1:i2,j1:j2] += R_heat/c_C;

  I_in = None; # signal not relevant for noise
  I1_q = 0; I2_q = I1_q + num_conc_q;
  I1_theta = I2_q; I2_theta = I1_theta + num_conc_theta;
  I_local_in = None; # signal not relevant for noise 
  I_local_out = {'i1':[I1_q,I1_theta],'i2':[I2_q,I2_theta]};  
  I_out = {'i1':[I1_conc_q,I1_conc_theta],'i2':[I2_conc_q,I2_conc_theta]};

  B_j = sqrt_two_k_B*B_j; # scale so BB^T = 2*k_B*K 

  B_j_indexing = {'I_in':I_in,'I_out':I_out,
                  'I_local_in':I_local_in,'I_local_out':I_local_out};

  return B_j,B_j_indexing; 

def compute_B_j__interface(Y,params,extras): 

  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));
  num_mesh_x,num_mesh_y,deltaX = tuple(map(
    params.get,['num_mesh_x','num_mesh_y','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX; 
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  kappa_P_I,kappa_C_I = tuple(map(
    params.get,['kappa_P_I','kappa_C_I']));
  num_dim = params['num_dim']; 
  num_dim_sq = num_dim*num_dim; 
  k_B = params['k_B'];
  sqrt_two_k_B = np.sqrt(2.0*k_B);

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_theta, = tuple(map(
    dd.get,['particle_theta']));
  conc_theta, = tuple(map(
    dd.get,['conc_theta']));
  interface_theta, = tuple(map(
    dd.get,['interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_theta,I2_particle_theta = tuple(map(
   dd.get,['I1_particle_theta','I2_particle_theta']));
  I1_conc_theta,I2_conc_theta = tuple(map(
    dd.get,['I1_conc_theta','I2_conc_theta']));
  I1_interface_theta,I2_interface_theta = tuple(map(
    dd.get,['I1_interface_theta','I2_interface_theta']));

  num_particle_theta = particle_theta.shape[0]; 
  num_conc_theta = conc_theta.shape[0]; 
  num_interface_theta = interface_theta.shape[0];

  if extras is not None:
    bar_K_j,bar_K_j_indices,matrix_vec_div,flag_save = \
      tuple(map(extras.get,['bar_K_j','bar_K_j_indices',
            'matrix_vec_div','flag_save']));
    extras_matrix_vec_div,extras_K_ovdc_dot_G = \
      tuple(map(extras.get,['extras_matrix_vec_div',
                            'extras_K_ovdc_dot_G']));
    matrix_Lambda_op, = tuple(map(extras.get,
                        ['matrix_Lambda_op']));
  else:
    bar_K_j,bar_K_j_indices,matrix_vec_div,flag_save = \
      (None,None,None,None);
    extras_matrix_vec_div,extras_K_ovdc_dot_G = \
      (None,None);
    matrix_Lambda_op = None;

  if flag_save is None:
    flag_save = False; 

  extras_R_ovdc = {};
  if extras_matrix_vec_div is not None:
    extras_R_ovdc.update({'matrix_vec_div':extras_matrix_vec_div['D'],
                          'extras_K_ovdc_dot_G':extras_K_ovdc_dot_G});
    matrix_vec_div = extras_matrix_vec_div['D'];

  # setup spatial distribution for the temperature coupling 
  if matrix_Lambda_op is None:
    matrix_Lambda_op = compute_matrix_Lambda_op(Y,params);

  Lambda_scalar_op = matrix_Lambda_op[0::num_dim,0]; # get the scalar component of Lambda operator

  kappa_C_I_xx_dx = kappa_C_I*Lambda_scalar_op;  # spatial dependence of thermal conductivity

  # compute interface tensor 
  ii = c_v_I; 
  j = ii['interface']; c_I = c_v[j];

  n1 = num_particle_theta + num_conc_theta + num_interface_theta;
  n2 = num_particle_theta + num_conc_theta + num_interface_theta;  
 
  i1_particle_theta = 0; 
  i2_particle_theta = i1_particle_theta + num_particle_theta; 
  i1_conc_theta = i2_particle_theta; 
  i2_conc_theta = i1_conc_theta + num_conc_theta;
  i1_interface_theta = i2_conc_theta; 
  i2_interface_theta = i1_interface_theta + num_interface_theta; 

  i1_B_theta_P = j1_B_theta_P = 0; 
  i1_B_theta_C = j1_B_theta_C = num_particle_theta;
  i1_B_theta_I = j1_B_theta_I = num_particle_theta + num_conc_theta;

  B_j = np.zeros((n1,n2));
  
  # --
  # heat exchange terms for P_I
  theta_P = particle_theta; theta_C = conc_theta; theta_I = interface_theta;
  ii = c_v_I;
  c_P = c_v[ii['particle']]; c_C = c_v[ii['conc']]; c_I = c_v[ii['interface']];

  i1_R21_theta_P = 0;
  i1_R21_theta_I = i1_R21_theta_P + num_particle_theta;

  nn_theta = num_particle_theta + num_interface_theta
  R_21 = np.zeros((nn_theta,num_particle_theta)); # is just a vector 

  R_21[0,0] = np.sqrt(kappa_P_I*theta_P[0]*theta_I[0])*(1.0/c_P); 
  R_21[1,0] = np.sqrt(kappa_P_I*theta_P[0]*theta_I[0])*(-1.0/c_I); 
  
  # copy R_21 into B_j 
  i1 = i1_B_theta_P; i2 = i1 + num_particle_theta;
  j1 = j1_B_theta_P; j2 = j1 + num_particle_theta;
  ii1 = i1_R21_theta_P; ii2 = ii1 + num_particle_theta; 
  jj1 = 0; jj2 = jj1 + num_particle_theta;
  B_j[i1:i2,j1:j2] = R_21[ii1:ii2,jj1:jj2];  

  i1 = i1_B_theta_I; i2 = i1 + num_interface_theta;
  j1 = j1_B_theta_P; j2 = j1 + num_particle_theta;
  ii1 = i1_R21_theta_I; ii2 = ii1 + num_interface_theta; 
  jj1 = 0; jj2 = jj1 + num_particle_theta;
  B_j[i1:i2,j1:j2] = R_21[ii1:ii2,jj1:jj2];  
  
  # heat exchange terms for C_I
  num_theta = 2;
  i1_R22_theta_C = 0; i1_R22_theta_I = 1;
  R_22_x = np.zeros((num_theta,num_mesh_pts)); # is just a two vector for each x location
  R_22_x[i1_R22_theta_C,:] = np.sqrt(kappa_C_I_xx_dx*theta_C*theta_I)*(1.0/(c_C*deltaV)); # scaling by deltaV 
  R_22_x[i1_R22_theta_I,:] = -np.sqrt(kappa_C_I_xx_dx*theta_C*theta_I)*(1.0/c_I); # set last row

  # copy into B_j 
  i1 = i1_B_theta_C; i2 = i1 + num_conc_theta;
  ii_range = range(i1,i2); 
  j1 = j1_B_theta_C; j2 = j1 + num_conc_theta;
  jj_range = range(j1,j2);
  B_j[ii_range,jj_range] += R_22_x[i1_R22_theta_C,:];

  ii_range = i1_B_theta_I*np.ones(num_conc_theta,dtype=int);
  j1 = j1_B_theta_C; j2 = j1 + num_conc_theta;
  jj_range = range(j1,j2);
  B_j[ii_range,jj_range] += R_22_x[i1_R22_theta_I,:];  
 
  I_in = None; 
  I_local_in = None; 
  I1_theta1 = 0; I2_theta1 = I1_theta1 + num_particle_theta;
  I1_theta2 = I2_theta1; I2_theta2 = I1_theta2 + num_conc_theta;
  I1_theta3 = I2_theta2; I2_theta3 = I1_theta3 + num_interface_theta;
  I_local_out = {'i1':[I1_theta1,I1_theta2,I1_theta3],
                 'i2':[I2_theta1,I2_theta2,I2_theta3]};  
  I_out = {'i1':[I1_particle_theta,I1_conc_theta,I1_interface_theta],
           'i2':[I2_particle_theta,I2_conc_theta,I2_interface_theta]};

  B_j = sqrt_two_k_B*B_j; # scale so BB^T = 2*k_B*K 

  B_j_indexing = {'I_in':I_in,'I_out':I_out,
                  'I_local_in':I_local_in,'I_local_out':I_local_out};

  return B_j,B_j_indexing; 

def compute_B_j_factors__conc(Y,params=None,extras=None):
  
  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));

  B_j_list = []; 
  I_in_list = []; I_out_list = []; 
  I_local_in_list = []; I_local_out_list = []; 
  ii = c_v_I;

  # ==
  # particle factors 
  # --
  B_j,B_j_indexing = compute_B_j__particle(Y,params,extras); 

  B_j_list.append(B_j); 
  I_in_list.append(B_j_indexing['I_in']); 
  I_out_list.append(B_j_indexing['I_out']); 
  I_local_in_list.append(B_j_indexing['I_local_in']); 
  I_local_out_list.append(B_j_indexing['I_local_out']);

  # ==
  # conc field factors
  # --
  B_j,B_j_indexing = compute_B_j__conc_field(Y,params,extras); 

  B_j_list.append(B_j); 
  I_in_list.append(B_j_indexing['I_in']); 
  I_out_list.append(B_j_indexing['I_out']); 
  I_local_in_list.append(B_j_indexing['I_local_in']); 
  I_local_out_list.append(B_j_indexing['I_local_out']);

  # ==
  # interface factors
  # --
  B_j,B_j_indexing = compute_B_j__interface(Y,params,extras); 

  B_j_list.append(B_j); 
  I_in_list.append(B_j_indexing['I_in']); 
  I_out_list.append(B_j_indexing['I_out']); 
  I_local_in_list.append(B_j_indexing['I_local_in']); 
  I_local_out_list.append(B_j_indexing['I_local_out']);

  # == 
  B_j_indices = {'I_in':I_in_list,'I_out':I_out_list,
                 'I_local_in':I_local_in_list,'I_local_out':I_local_out_list};

  return B_j_list,B_j_indices; 

def compute_g_thm_j_dt__conc(Y_n,params,extras=None):
  # Compute fluctuation constributions.

  num_dim,k_B,Y_I,deltaT,flag_save_B_j_tensors,flag_compute_div_K = \
    tuple(map(params.get,['num_dim','k_B','Y_I','deltaT',
                          'flag_save_B_j_tensors','flag_compute_div_K']));

  if num_dim is None or k_B is None or Y_I is None or deltaT is None \
     or flag_compute_div_K is None:
    ss = 'num_dim,k_B,Y_I,deltaT,flag_save_B_j_tensors,flag_compute_div_K';
    raise Exception("Missing one of the parameters in params dict: " + ss); 

  if extras is not None:
    flag_save_B_j_tensors,flag_save_div_K_j,flag_save_dW = \
      tuple(map(extras.get,['flag_save_B_j_tensors',
          'flag_save_div_K_j','flag_save_dW']));
    flag_use_saved_dW, = \
      tuple(map(extras.get,['flag_use_saved_dW']));
  else:
    flag_save_B_j_tensors,flag_save_div_K_j, \
    flag_save_dW,flag_use_saved_dW = \
      (None,None,None,None);

  if flag_save_B_j_tensors is None:
    flag_save_B_j_tensors = False;

  if flag_save_div_K_j is None:
    flag_save_div_K_j = False;

  if flag_save_dW is None:
    flag_save_dW = False;

  if flag_use_saved_dW is None:
    flag_use_saved_dW = False;

  B_j_list,B_j_indices = compute_B_j_factors__conc(Y_n,params,extras); ii = Y_I;

  if flag_compute_div_K:
    div_K_j_list = compute_div_K_j__conc(Y_n,params,extras);
    if flag_save_div_K_j:
      extras.update({'div_K_j_list':div_K_j_list});

  if flag_save_B_j_tensors:
    extras.update({'B_j':B_j_list,'B_j_indices':B_j_indices});
    if flag_compute_div_K:
      extras.update({'div_K_j_list':div_K_j_list});

  if flag_use_saved_dW:
    dW_list = extras['dW_list'];
  elif flag_save_dW:
    dW_list = [];
    
  g_thm_j_dt_list = [];
  sqrt_deltaT = np.sqrt(deltaT);
  num_heat_bodies = params['num_heat_bodies']; # particle, fluid, interface 

  # generate stochastic forces
  for j in range(0,num_heat_bodies): # @ optimize
    B_j = B_j_list[j];
    num_in = B_j.shape[1]; num_out = B_j.shape[0];

    # stochastic part of g_thm
    if flag_use_saved_dW is False:
      xi = np.random.randn(num_in);
      dW = xi*sqrt_deltaT; # increments Wiener process
    else: # use saved values (needed by some SDE numerical methods)
      dW = dW_list[j];
 
    # generate stochastic driving field
    B_j__dW = np.dot(B_j,dW);
    g_thm_dt = B_j__dW;
   
    if flag_save_dW and (flag_use_saved_dW is False):
      dW_list.append(dW);
    
    # divergence part of g_thm
    if flag_compute_div_K:
      # (need to make sure same indexing as for B_j_indices)
      div_K_j = div_K_j_list[j];
      g_thm_dt += k_B*div_K_j*deltaT;

    g_thm_j_dt_list.append(g_thm_dt);

  if flag_save_dW and (flag_use_saved_dW is False):
    extras.update({'dW_list':dW_list});

  return g_thm_j_dt_list,B_j_indices;

