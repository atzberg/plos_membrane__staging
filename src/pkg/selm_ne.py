#!/usr/bin/env python
#
# coding: utf-8
#
# SELM Non-equilibrium.
#
# Continuum Fluid Heat Body, Concentration Fields, Temperature Fields
#
# Discrete Particles and Interfacial Regions
# 
#
# We build in part on Mielke 2011 paper, and GENERIC framework with 
# $\bar{K} = N_\mathcal{E} K_0 N_\mathcal{E} since this gives us a natural
# way to handle the continuous fields and heat fluxes. 
#
# We also perform other derivations and formulations to obtain overdamped
# results and other regimes.  We also perform analysis and derivations 
# to obtain discretizations with good properties.  We also perform analysis
# for obtaining algorithms to generate the stochastic fields. 
#
#
#
#

print("="*80);

# -- Imports
import numpy as np
import pickle;
import time; 
import os; 
import sys;
import ipdb;
import shutil;

import argparse; 

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk

def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);   

# =============================
# debugging helper functions
# =============================
def print_dict(dd):
  for k in dd.keys():
    print("%-20s:%s"%(str(k),str(dd[k])));

def print_matrix(M,name="",fm='%+7.2e'):
  """
  Prints a matrix and formats.  
  Some suggested formats include 
  fm="%0.1g", fm="%d", fm="0.1e".
  """
  # for debugging

  if name == "":
    print(""); 
  else:
    print(name + " = ");
  ss = "";
  for i in range(0,M.shape[0]):
    if i == 0:
      ss += "[";
    for j in range(0,M.shape[1]):
      if j > 0:
        ss += "  ";
      else:
        if i > 0:
          ss += " [";
        else:
          ss += "[";  
      ss += fm%M[i,j];
    ss += " ]";
    if i == M.shape[0] - 1:
      ss += "]";
    ss += "\n";
    
  print(ss);

# wrap debug
# dd = lambda tt,params=params,vars=locals(): debug_to_csv(params,tt,vars[tt])
def debug_to_csv(params,name,tensor):
  filename = params['debug_dir'] + '/' + name + '.csv';
  print("filename = " + filename); 
  np.savetxt(filename, tensor, delimiter=",");

# alias
p_m = print_matrix;
p_d = print_dict; 

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

def proto4vtr(params):
  xx = get_mesh_xx(params);
  ff = {};
  ff['field_values'] = None;
  ff['field_name'] = 'debug';
  ff['NumberOfComponents'] = 1;
  fff = [ff];
  filename = params['base_dir'] + '/debug/debug1.vtr';
  return xx,fff,filename;

def sup_norm(a):
  return np.max(np.abs(a.flatten()));

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


# =============================
def write_vtr_data(filename,xx,field_list,flag_verbose=1):

  #ipdb.set_trace();

  # create the unstructured grid
  vtr_grid = vtk.vtkRectilinearGrid();

  # num per direction
  vtr_grid.SetDimensions(xx[0].shape[0],
                         xx[1].shape[0],
                         xx[2].shape[0]);

  num_points = 1;
  for d in range(0,3):
    num_points *= xx[d].shape[0];
     
  # setup the general grid
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
    # WARNING: converting values to process in vtr.
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
      #f_data.SetVoidArray(field_values, num_points, num_components);
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
      #f_data.SetVoidArray(field_values.flatten(), num_points*num_dim, num_components);
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

  # write the unstructured grid to XML file
  vtr_writer = vtk.vtkXMLRectilinearGridWriter();
  vtr_writer.SetFileName(filename);
  vtr_writer.SetInputData(vtr_grid);
  vtr_writer.SetCompressorTypeToNone(); # help ensure ascii output (as opposed to binary)
  vtr_writer.SetDataModeToAscii(); # help ensure ascii output (as opposed to binary)
  vtr_writer.Write();


def write_vtp_data(vtp_filename,points,field_list,flag_verbose=1):
  # output a VTP file with the fields

  # record the data for output
  vtp_data = vtk.vtkPolyData();

  # setup the points data
  vtp_points = vtk.vtkPoints();
  ambient_num_dim = points.shape[1]; 
  num_points = points.shape[0];
  for I in range(num_points):
    vtp_points.InsertNextPoint(points[I,0],points[I,1],points[I,2]);
  
  vtp_data.SetPoints(vtp_points);
 
  # Get data from the vtu object
  #print("Getting data from the vtu object.");
  #nodes_vtk_array = vtuData.GetPoints().GetData();
  #ptsX            = vtk_to_numpy(nodes_vtk_array);

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


# =============================
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

# alias
write_a = write_array_vtr; 

def get_comp(Y,i1_str,i2_str,Y_I):
  return Y[Y_I[i1_str]:Y_I[i2_str]];

def set_comp(Y,i1_str,i2_str,Y_I,val):
  Y[Y_I[i1_str]:Y_I[i2_str]] = val;


def add_in_components(Y_out,a_out,I_out,I_local_out): 
  ii1_list = I_out['i1']; ii2_list = I_out['i2']; # output indices
  i1_list = I_local_out['i1']; i2_list = I_local_out['i2']; # internal indices
  for k in range(0,len(i1_list)): # transfer results into correct components of Y
    i1 = i1_list[k]; i2 = i2_list[k];
    ii1 = ii1_list[k]; ii2 = ii2_list[k];
    Y_out[ii1:ii2] += a_out[i1:i2];

def add_in_matrix_entries(A,B,I_out,I_in,I_local_out,I_local_in): 
  ii1_list = I_out['i1']; ii2_list = I_out['i2']; # output indices
  jj1_list = I_in['i1']; jj2_list = I_in['i2']; # output indices
  i1_list = I_local_out['i1']; i2_list = I_local_out['i2']; # internal indices
  j1_list = I_local_in['i1']; j2_list = I_local_in['i2']; # internal indices
  for k in range(0,len(i1_list)): # transfer results into correct entries of A
    i1 = i1_list[k]; i2 = i2_list[k];
    j1 = j1_list[k]; j2 = j2_list[k];
    ii1 = ii1_list[k]; ii2 = ii2_list[k];
    jj1 = jj1_list[k]; jj2 = jj2_list[k];
    A[ii1:ii2,jj1:jj2] += B[i1:i2,j1:j2];

def apply_tensors(Y_out,A_list,I_out_list,I_local_out_list):
  pass;


def get_parts_I(params):
  return params['Y_I'];

def map_particle_periodic(Y,params):
  """ Map particles mod L to the periodic domain"""

  Y_I = params['Y_I']; num_dim = params['num_dim'];
  #locals().update(**Y_I); # adds indices to local variables 
  num_mesh_x,num_mesh_y,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','deltaX']));
  Lx = num_mesh_x*deltaX; Ly = num_mesh_y*deltaX;

  I1_particle_q,I2_particle_q = tuple(map(
    Y_I.get,['I1_particle_q','I2_particle_q']));

  # hard-coded num_dim = 2
  if num_dim != 2:
    raise Exception("implemented currently only for num_dim == 2");
 
  Y[I1_particle_q + 0] = np.mod(Y[I1_particle_q + 0],Lx);
  Y[I1_particle_q + 1] = np.mod(Y[I1_particle_q + 1],Ly);
  
def debug_projection(Y,params): 
  print("."*80);
  print("debugging projection (wp matrix):");
  print("compute_matrix_wp");
  matrix_wp = compute_matrix_wp(Y,params);
  print("matrix_wp.shape " + str(matrix_wp.shape));

  Y_I = params['Y_I'];
  num_fluid_p = Y_I['I2_fluid_p'] - Y_I['I1_fluid_p'];
  num_dim = params['num_dim'];

  flow_index = 0;

  # ==
  # write the data to .vtr files to qualitatively check 
  debug_dir = params['debug_dir'];
  xx = get_mesh_xx(params);
  filename = debug_dir + '/wp_matrix.vtr';
  fff=[];

  # ..
  # apply the projection matrix to a few velocity fields 
  
  # flow
  fluid_p = np.zeros(num_fluid_p);
  x1,x2,Lx,Ly = get_mesh_coord(params); k1 = 1.0; k2 = 1.0;

  fluid_p[0::num_dim] = 0.1*np.sin(2.0*np.pi*k1*x1/Lx)*np.sin(2.0*np.pi*k2*x2/Ly); 
  fluid_p[1::num_dim] = 0.2*np.sin(2.0*np.pi*k1*x2/Lx)*np.sin(2.0*np.pi*k2*x2/Ly); 

  fluid_p_wp = np.matmul(matrix_wp,fluid_p);
   
  ff = {};
  ff['field_name'] = 'v_flow_%.2d'%flow_index;
  ff['NumberOfComponents'] = 3;
  fluid_pp = conv_2d_vec_3d(fluid_p,num_dim);
  ff['field_values'] = fluid_pp; 
  fff.append(ff);

  ff = {};
  ff['field_name'] = 'v_flow_wp_%.2d'%flow_index;
  ff['NumberOfComponents'] = 3;
  fluid_pp = conv_2d_vec_3d(fluid_p_wp,num_dim);
  ff['field_values'] = fluid_pp; 
  fff.append(ff);
  
  flow_index += 1; 

  # flow
  fluid_p = np.zeros(num_fluid_p);
  x1,x2,Lx,Ly = get_mesh_coord(params); k1 = 2.0; k2 = 3.0;

  fluid_p[0::num_dim] = 0.3*np.sin(2.0*np.pi*k1*x1/Lx)*np.sin(2.0*np.pi*k2*x2/Ly); 
  fluid_p[1::num_dim] = 0.3*np.cos(2.0*np.pi*k1*x2/Lx)*np.sin(2.0*np.pi*k2*x2/Ly); 

  fluid_p_wp = np.matmul(matrix_wp,fluid_p);
   
  ff = {};
  ff['field_name'] = 'v_flow_%.2d'%flow_index;
  ff['NumberOfComponents'] = 3;
  fluid_pp = conv_2d_vec_3d(fluid_p,num_dim);
  ff['field_values'] = fluid_pp; 
  fff.append(ff);

  ff = {};
  ff['field_name'] = 'v_flow_wp_%.2d'%flow_index;
  ff['NumberOfComponents'] = 3;
  fluid_pp = conv_2d_vec_3d(fluid_p_wp,num_dim);
  ff['field_values'] = fluid_pp; 
  fff.append(ff);
  
  flow_index += 1; 

  # write the data
  print("filename = " + filename); 
  write_vtr_data(filename,xx,fff);

  # --
  # compute the numerical divergence of the fields (and report)
  flag_exit = False;
  if flag_exit:
    print("exit()");
    print("");
    sys.exit(); 


def compute_matrix_dft_2d_mesh_scalar(Y,params):
  #I1_fluid_p = params['Y_I']['I1_fluid_p'];
  #I2_fluid_p = params['Y_I']['I2_fluid_p'];
  #num_dim = params['num_dim'];
  #num_fluid_p = I2_fluid_p - I1_fluid_p;
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  num_mesh_pts = num_mesh_x*num_mesh_y;

  #matrix_dft_x = np.fft.fft(np.eye(num_mesh_x));
  #matrix_dft_y = np.fft.fft(np.eye(num_mesh_y));
  
  probe_flat = np.eye(num_mesh_pts);
  probe = probe_flat.reshape((num_mesh_pts,num_mesh_x,num_mesh_y));
  
  matrix_dft_2 = np.fft.fft2(probe); 
  
  matrix_dft_2_flat = matrix_dft_2.reshape((num_mesh_pts,num_mesh_pts));
  
  return matrix_dft_2_flat; 

def compute_matrix_idft_2d_mesh_scalar(Y,params):
  #I1_fluid_p = params['Y_I']['I1_fluid_p'];
  #I2_fluid_p = params['Y_I']['I2_fluid_p'];
  #num_dim = params['num_dim'];
  #num_fluid_p = I2_fluid_p - I1_fluid_p;
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  num_mesh_pts = num_mesh_x*num_mesh_y;

  #matrix_dft_x = np.fft.fft(np.eye(num_mesh_x));
  #matrix_dft_y = np.fft.fft(np.eye(num_mesh_y));
  
  probe_flat = np.eye(num_mesh_pts);
  probe = probe_flat.reshape((num_mesh_pts,num_mesh_x,num_mesh_y));
  
  matrix_idft_2 = np.fft.ifft2(probe); 
  
  matrix_idft_2_flat = matrix_idft_2.reshape((num_mesh_pts,num_mesh_pts));
  
  return matrix_idft_2_flat; 

def compute_matrix_dft_2d_mesh_vec(Y,params):
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  num_mesh_pts = num_mesh_x*num_mesh_y;

  matrix_dft_2_flat_scalar = compute_matrix_dft_2d_mesh_scalar(Y,params); 
  matrix_dft_2_vec = np.zeros((num_mesh_pts,num_dim,num_mesh_pts,num_dim),dtype=np.complex64); 
  matrix_dft_2_vec[:,0,:,0] = matrix_dft_2_flat_scalar;
  matrix_dft_2_vec[:,1,:,1] = matrix_dft_2_flat_scalar;
  matrix_dft_2_vec_flat = matrix_dft_2_vec.reshape((num_mesh_pts*num_dim,num_mesh_pts*num_dim)); 

  return matrix_dft_2_vec_flat;
 
def compute_matrix_idft_2d_mesh_vec(Y,params):
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  num_mesh_pts = num_mesh_x*num_mesh_y;

  matrix_idft_2_flat_scalar = compute_matrix_idft_2d_mesh_scalar(Y,params); 
  matrix_idft_2_vec = np.zeros((num_mesh_pts,num_dim,num_mesh_pts,num_dim),dtype=np.complex64); 
  matrix_idft_2_vec[:,0,:,0] = matrix_idft_2_flat_scalar;
  matrix_idft_2_vec[:,1,:,1] = matrix_idft_2_flat_scalar;
  matrix_idft_2_vec_flat = matrix_idft_2_vec.reshape((num_mesh_pts*num_dim,num_mesh_pts*num_dim)); 

  return matrix_idft_2_vec_flat;
 

def compute_matrix_project_vec_mode(Y,params):
  # double-check below calculation for the projection operations 

  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  num_mesh_pts = num_mesh_x*num_mesh_y;
 
  # projection operator is $\hat{p}_k = (I - \frac{\mb{g}_k\mb{g}_k^T}{\|\mb{g}_k\|}^2)$ 
  kk1 = np.linspace(0,num_mesh_x-1,num_mesh_x); kk2 = np.linspace(0,num_mesh_y-1,num_mesh_y);
  kk = np.meshgrid(kk1,kk2); 
  vec_k = np.zeros((num_mesh_x,num_mesh_y,num_dim)); 
  vec_k[:,:,0] = kk[1]; vec_k[:,:,1] = kk[0];

  g_k = np.zeros((num_mesh_x,num_mesh_y,num_dim));  
  norm_vec_k = np.expand_dims(np.sqrt(np.sum(vec_k*vec_k,2)),2);
  
  g_k[0::,1::,:] = vec_k[0::,1::,:]/norm_vec_k[0::,1::,:];
  g_k[1::,0,:] = vec_k[1::,0,:]/norm_vec_k[1::,0,:];

  g1_k = np.expand_dims(g_k,3); 
  g2_k = np.expand_dims(g_k,2);
  g_k_g_k_T = g1_k*g2_k; # using broadcast rules  

  I_k = np.zeros((num_mesh_x,num_mesh_y,num_dim,num_dim)); 
  I = np.eye(num_dim); 
  II = np.expand_dims(I,(0,1));
  I_k[:,:,:,:] = II; # using broadcast rules

  II_k = I_k.reshape((num_mesh_pts,num_dim,num_dim)); 
  gg_k_gg_k_T = g_k_g_k_T.reshape((num_mesh_pts,num_dim,num_dim));

  mproj_raw = np.zeros((num_mesh_pts,num_dim,num_mesh_pts,num_dim));
  ii1 = range(0,num_mesh_pts); 
  mproj_raw[ii1,:,ii1,:] =  II_k - gg_k_gg_k_T; 

  mproj = mproj_raw.reshape(num_mesh_pts*num_dim,num_mesh_pts*num_dim);

  return mproj; 
  
   
def compute_matrix_wp(Y,params,extras=None):

  if extras is not None:
    matrix_wp,flag_save = tuple(map(extras.get,['matrix_wp','flag_save']));
  else:
    matrix_wp = None;
    flag_save = None; 

  if flag_save is None:
    flag_save = False; 

  if matrix_wp is not None: # no need to re-compute, just return it
    return matrix_wp;

  # @@@ double-check below for projections 

  # dft of vector field 
  matrix_dft_2_vec = compute_matrix_dft_2d_mesh_vec(Y,params); 

  # perform the projection 
  matrix_project_vec_mode = compute_matrix_project_vec_mode(Y,params); 

  # map back to vector field 
  matrix_idft_2_vec = compute_matrix_idft_2d_mesh_vec(Y,params);

  # compose the projection 
  matrix_wp = np.matmul(matrix_idft_2_vec,np.matmul(matrix_project_vec_mode,matrix_dft_2_vec));

  if flag_save:
    extras['matrix_wp'] = matrix_wp;

  return matrix_wp;  




def compute_D_j_interface(Y,params,extras=None):
  """
  Dissipation at the fluid-structure interface.  Note, one would need one 
  temperature for each distinct micro-structure.  This would then correspond to 
  having a dissipative operator for each of the micro-structures. 
  """

  m,rho,gamma,mu,Y_I,c_v,c_v_I,deltaX = tuple(map(
    params.get,['m','rho','gamma','mu','Y_I','c_v','c_v_I','deltaX']));
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
  particle_q,particle_p,particle_theta = tuple(map(dd.get,['particle_q','particle_p','particle_theta']));
  fluid_phi,fluid_p,fluid_theta = tuple(map(dd.get,['fluid_phi','fluid_p','fluid_theta']));
  interface_q,interface_p,interface_theta = tuple(map(dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  # locals().update(**Y_I)  # quick way to make dict elements local variables (not best practice though)
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_p,I2_particle_p = tuple(map(dd.get,['I1_particle_p','I2_particle_p']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_fluid_phi,I2_fluid_phi = tuple(map(dd.get,['I1_fluid_phi','I2_fluid_phi']));
  I1_fluid_p,I2_fluid_p = tuple(map(dd.get,['I1_fluid_p','I2_fluid_p']));
  I1_fluid_theta,I2_fluid_theta = tuple(map(dd.get,['I1_fluid_theta','I2_fluid_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_p = particle_p.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_fluid_phi = fluid_phi.shape[0]; num_fluid_p = fluid_p.shape[0]; num_fluid_theta = fluid_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_p = interface_p.shape[0]; num_interface_theta = interface_theta.shape[0];

  num_q = num_particle_q + num_fluid_phi; num_p = num_particle_p + num_fluid_p;
  ii1_particle_p = 0; ii2_particle_p = ii1_particle_p + num_particle_p;
  ii1_fluid_p = ii2_particle_p; ii2_fluid_p = ii1_fluid_p + num_fluid_p;
  #D_j = np.zeros((num_p, num_p)); 
  D_j = np.zeros((num_p, num_p));  # assumes only one temperature to track accumulated dissipation energy
 
  # particle drag term from the interface 
  D_j[ii1_particle_p:ii2_particle_p,ii1_particle_p:ii2_particle_p] = gamma*np.eye(num_particle_p);
  matrix_Gamma_op = compute_matrix_Gamma_op(Y,params); 
  D_j[ii1_particle_p:ii2_particle_p,ii1_fluid_p:ii2_fluid_p] = -gamma*matrix_Gamma_op/deltaV; # factor deltaV given energy density for fluid 

  # fluid drag term from the interface 
  D_j[ii1_fluid_p:ii2_fluid_p,ii1_particle_p:ii2_particle_p] = np.transpose(D_j[ii1_particle_p:ii2_particle_p,ii1_fluid_p:ii2_fluid_p]);
  D_j[ii1_fluid_p:ii2_fluid_p,ii1_fluid_p:ii2_fluid_p] = gamma*np.dot(matrix_Gamma_op.T,matrix_Gamma_op)/(deltaV*deltaV); # factor deltax_sq given energy density for fluid # @@@ check 

  if flag_save:
    extras.update({'matrix_Gamma_op':matrix_Gamma_op});

  return D_j;

def compute_D_j_particle(Y,params,extras=None):
  m,rho,gamma,mu,Y_I,c_v,c_v_I = tuple(map(params.get,['m','rho','gamma','mu','Y_I','c_v','c_v_I']));
  num_particles,num_dim = tuple(map(params.get,['num_particles','num_dim']));

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_p,particle_theta = tuple(map(dd.get,['particle_q','particle_p','particle_theta']));
  fluid_phi,fluid_p,fluid_theta = tuple(map(dd.get,['fluid_phi','fluid_p','fluid_theta']));
  interface_q,interface_p,interface_theta = tuple(map(dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  # locals().update(**Y_I)  # quick way to make dict elements local variables (not best practice though)
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_p,I2_particle_p = tuple(map(dd.get,['I1_particle_p','I2_particle_p']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_p = particle_p.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_fluid_phi = fluid_phi.shape[0]; num_fluid_p = fluid_p.shape[0]; num_fluid_theta = fluid_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_p = interface_p.shape[0]; num_interface_theta = interface_theta.shape[0];

  num_q = num_particle_q; num_p = num_particle_p;
  #D_j = np.zeros((num_p, num_p));   
  D_j = 0*np.eye(num_p);  

  return D_j;



# def compute_M_E_j(Y,params):
#   m,gamma,mu_1,mu_2,Y_I,c_v = tuple(map(params.get,['m','gamma','mu_1','mu_2','Y_I','c_v']));
#   q,p,theta = get_parts(Y,params);
#   I1_q,I2_q,I1_p,I2_p,I1_theta,I2_theta = get_parts_I(params);
#   num_q = q.shape[0]; num_p = p.shape[0];
#   num_heat_bodies = c_v.shape[0];
#   M_E_j = np.zeros((num_heat_bodies, num_q + num_p + 1, num_q + num_p + 1)); 
#   for j in range(0,num_heat_bodies):
#     M_E_j[j,I1_q:I2_q,I1_q:I2_q] = np.eye(I2_q - I1_q); 
#     M_E_j[j,I1_p:I2_p,I1_p:I2_p] = np.eye(I2_p - I1_p); 
#     M_E_j[j,I2_p + 1,I1_q:I2_q] = -grad_q_u_j_T/partial_theta_j_u_j; 
#     M_E_j[j,I2_p + 1,I1_p:I2_p] = -p_T/partial_theta_j_u_j; 
#     M_E_j[j,I1_p + 1,I2_p + 1] = 1.0/partial_theta_j_u_j; 
# 
#   return M_E_j;

def compute_matrix_tensor_div(Y,params,extras=None):
  """ Divergence acting on tensor $\sigma$ to produce vector $b$, 
      $b = div(\sigma),\; b_i = \partial_{j} \sigma_{ij}$.
  """
  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX \
    = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));  
  num_dim_sq = num_dim*num_dim; # tensor dimension 

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
  D = np.zeros((num_mesh_x*num_mesh_y*num_dim,num_mesh_x*num_mesh_y*num_dim_sq));
  
  # @ optimize 
  vec_I = np.zeros(num_dim,dtype=int); vec_num_mesh = np.zeros(num_dim,dtype=int);
  vec_num_mesh[0] = num_mesh_x; vec_num_mesh[1] = num_mesh_y;
  J0_mesh_iim1 = np.zeros(num_dim,dtype=int); J0_mesh_iip1 = np.zeros(num_dim,dtype=int); 
  for j in range(0,num_mesh_y): 
    for i in range(0,num_mesh_x):
      vec_I[0] = i; vec_I[1] = j; 
      vec_Im1 = np.zeros(vec_I.shape,dtype=int); vec_Ip1 = np.zeros(vec_I.shape,dtype=int); # same shape arrays
      I0_mesh = vec_I[1]*num_mesh_x*num_dim + vec_I[0]*num_dim + 0;
      for d in range(0,num_dim):
        vec_Im1[:] = vec_I[:]; vec_Ip1[:] = vec_I[:]; # copy index then perturb it only in index d 
        iim1 = vec_I[d] - 1; iip1 = vec_I[d] + 1;
        if iim1 < 0: 
          iim1 = vec_num_mesh[d] + iim1; # periodic
        if iip1 >= vec_num_mesh[d]: 
          iip1 = iip1 - vec_num_mesh[d]; # periodic
        vec_Im1[d] = iim1; vec_Ip1[d] = iip1; 
        
        J0_mesh_iim1[d] = vec_Im1[1]*num_mesh_x*num_dim_sq + vec_Im1[0]*num_dim_sq + 0;
        J0_mesh_iip1[d] = vec_Ip1[1]*num_mesh_x*num_dim_sq + vec_Ip1[0]*num_dim_sq + 0;

      for a in range(0,num_dim):
        for b in range(0,num_dim):
          D[I0_mesh + a,J0_mesh_iip1[b] + a*num_dim + b] += 1.0/(2.0*deltaX);
          D[I0_mesh + a,J0_mesh_iim1[b] + a*num_dim + b] += -1.0/(2.0*deltaX); 

  if flag_save: 
    extras['D'] = D;

  return D;


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

def compute_matrix_tensor_grad(Y,params,extras=None):
  """ Gradient acting on tensor fields, such as 
      $\nabla u$.
      We aim to construct this to be the 
      transpose of the divergence.  We could 
      just transpose the divergence matrix constructed. 
      We do a separate implementation to help verify 
      codes.  Can validate against the divergence matrix.  
  """
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
  G = np.zeros((num_mesh_x*num_mesh_y*num_dim_sq,num_mesh_x*num_mesh_y*num_dim));
  
  # @ optimize 
  vec_I = np.zeros(num_dim,dtype=int); vec_num_mesh = np.zeros(num_dim,dtype=int);
  vec_num_mesh[0] = num_mesh_x; vec_num_mesh[1] = num_mesh_y;
  J0_mesh_iim1 = np.zeros(num_dim,dtype=int); J0_mesh_iip1 = np.zeros(num_dim,dtype=int); 
  for j in range(0,num_mesh_y): 
    for i in range(0,num_mesh_x):
      vec_I[0] = i; vec_I[1] = j; 
      vec_Im1 = vec_I + 0; vec_Ip1 = vec_I + 0; # make copies
      I0_mesh = vec_I[1]*num_mesh_x*num_dim_sq + vec_I[0]*num_dim_sq + 0;
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

      for a in range(0,num_dim):
        for b in range(0,num_dim):
          G[I0_mesh + a*num_dim + b,J0_mesh_iip1[b] + a] += 1.0/(2.0*deltaX);
          G[I0_mesh + a*num_dim + b,J0_mesh_iim1[b] + a] += -1.0/(2.0*deltaX); 

  if flag_save:
    extras['G'] = G;

  return G;


def compute_matrix_tensor_grad2(Y,params,extras=None):
  """ Gradient acting on tensor fields with extra 
      averaging to help avoid the checkerboard instability. 
      Computes consistently $\nabla u$.
      We aim to construct this to be the 
      transpose of the divergence.  We could 
      just transpose the divergence matrix constructed. 
      We do a separate implementation to help verify 
      codes.  Can validate against the divergence matrix.  
  """
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
  num_mesh_pts = num_mesh_x*num_mesh_y;
  #G = np.zeros((num_mesh_pts*num_dim_sq,num_mesh_pts*num_dim));
  #GG = G.reshape(num_mesh_pts*num_dim_sq,num_mesh_x,num_mesh_y,num_dim);
  GG = np.zeros((num_mesh_pts*num_dim_sq,num_mesh_x,num_mesh_y,num_dim));

  Gdir = [];
  # x-derivative (scalar) 
  Gx = np.zeros((3,3));
  Gx[0,0] = -0.5; Gx[0,2] = -0.5;
  Gx[2,0] = 0.5; Gx[2,2] = 0.5;
  Gx = Gx/(2*deltaX);
  Gdir.append(Gx);

  # y-derivative (scalar)
  Gy = np.zeros((3,3));
  Gy[0,0] = -0.5; Gy[0,2] = 0.5;
  Gy[2,0] = -0.5; Gy[2,2] = 0.5;
  Gy = Gy/(2*deltaX);
  Gdir.append(Gy);

  # @ optimize 
  vec_I = np.zeros(num_dim,dtype=int);
  for j in range(0,num_mesh_y): 
    for i in range(0,num_mesh_x):
      vec_I[0] = i; vec_I[1] = j; 
      vec_Im1 = vec_I + 0; vec_Ip1 = vec_I + 0; # make copies
      I0_mesh = vec_I[1]*num_mesh_x*num_dim_sq + vec_I[0]*num_dim_sq + 0;

      for a in range(0,num_dim):
        for b in range(0,num_dim):
          # partial_b u_a
          for c1 in range(0,3):
            for c2 in range(0,3):
              ii = i + c1 - 1; jj = j + c2 - 1; 
              if ii < 0: ii = num_mesh_x + ii;
              if ii >= num_mesh_x: ii = ii - num_mesh_x;
              if jj < 0: jj = num_mesh_y + jj;
              if jj >= num_mesh_y: jj = jj - num_mesh_y;
              GG[I0_mesh + a*num_dim + b,ii,jj,a] = Gdir[b][c1,c2];

  G = GG.reshape((num_mesh_pts*num_dim_sq,num_mesh_pts*num_dim));

  if flag_save:
    extras['G'] = G;

  return G;

def compute_matrix_tensor_grad3(Y,params,extras=None):
  """ Gradient acting on tensor fields with extra 
      averaging to help avoid the checkerboard instability. 
      Combines usual central difference gradient with a 
      diagonal gradient stencil. 
  """
  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX \
    = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));  
  num_dim_sq = num_dim*num_dim; # tensor dimension 

  if num_dim != 2:
    raise Exception("assumes num_dim = 2, input gave num_dim = " + str(num_dim));

  if extras is not None:
    G,flag_save,alpha = tuple(map(extras.get,['G','flag_save','alpha']));
  else:
    G,alpha = (None,None);
    flag_save = None;

  if flag_save is None:
    flag_save = False; 

  if G is not None: # no need to re-compute, just return it
    return G;

  if alpha is None:
    alpha = 0.5; 

  if G is None:
    G1 = compute_matrix_tensor_grad(Y,params); 
    G2 = compute_matrix_tensor_grad2(Y,params); 
    G = (1 - alpha)*G1 + alpha*G2; 

  if flag_save:
    extras.update({'G':G});

  return G; 


def compute_matrix_vec_grad(Y,params,extras=None):
  """ Gradient acting on scalar fields, such as 
      $\nabla f$.
      We aim to construct this to be the 
      transpose of the divergence.  We could 
      just transpose the divergence matrix constructed. 
      We do a separate implementation to help verify 
      codes.  Can validate against the divergence matrix.  
  """
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

def compute_matrix_vec_grad2(Y,params,extras=None):
  """ Gradient acting on vec fields with extra 
      averaging to help avoid the checkerboard instability. 
      Computes consistently $\nabla u$.
      We aim to construct this to be the 
      transpose of the divergence.  We could 
      just transpose the divergence matrix constructed. 
      We do a separate implementation to help verify 
      codes.  Can validate against the divergence matrix.  
  """
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
  num_mesh_pts = num_mesh_x*num_mesh_y;
  #G = np.zeros((num_mesh_pts*num_dim_sq,num_mesh_pts*num_dim));
  #GG = G.reshape(num_mesh_pts*num_dim_sq,num_mesh_x,num_mesh_y,num_dim);
  GG = np.zeros((num_mesh_pts*num_dim,num_mesh_x,num_mesh_y));

  Gdir = [];
  # x-derivative (scalar) 
  Gx = np.zeros((3,3));
  Gx[0,0] = -0.5; Gx[0,2] = -0.5;
  Gx[2,0] = 0.5; Gx[2,2] = 0.5;
  Gx = Gx/(2*deltaX);
  Gdir.append(Gx);

  # y-derivative (scalar)
  Gy = np.zeros((3,3));
  Gy[0,0] = -0.5; Gy[0,2] = 0.5;
  Gy[2,0] = -0.5; Gy[2,2] = 0.5;
  Gy = Gy/(2*deltaX);
  Gdir.append(Gy);

  # @ optimize 
  vec_I = np.zeros(num_dim,dtype=int);
  for j in range(0,num_mesh_y): 
    for i in range(0,num_mesh_x):
      vec_I[0] = i; vec_I[1] = j; 
      vec_Im1 = vec_I + 0; vec_Ip1 = vec_I + 0; # make copies
      I0_mesh = vec_I[1]*num_mesh_x*num_dim + vec_I[0]*num_dim + 0;

      for b in range(0,num_dim):
        # partial_b u_a
        for c1 in range(0,3):
          for c2 in range(0,3):
            ii = i + c1 - 1; jj = j + c2 - 1; 
            if ii < 0: ii = num_mesh_x + ii;
            if ii >= num_mesh_x: ii = ii - num_mesh_x;
            if jj < 0: jj = num_mesh_y + jj;
            if jj >= num_mesh_y: jj = jj - num_mesh_y;
            GG[I0_mesh + b,ii,jj] = Gdir[b][c1,c2];

  G = GG.reshape((num_mesh_pts*num_dim,num_mesh_pts));

  if flag_save:
    extras['G'] = G;

  return G;


def compute_matrix_vec_grad3(Y,params,extras=None):
  """ Gradient acting on vec fields with extra 
      averaging to help avoid the checkerboard instability. 
      Combines usual central difference gradient with a 
      diagonal gradient stencil. 
  """
  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX \
    = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));  
  num_dim_sq = num_dim*num_dim; # vec dimension 

  if num_dim != 2:
    raise Exception("assumes num_dim = 2, input gave num_dim = " + str(num_dim));

  if extras is not None:
    G,flag_save,alpha = tuple(map(extras.get,['G','flag_save','alpha']));
  else:
    G,alpha = (None,None);
    flag_save = None;

  if flag_save is None:
    flag_save = False; 

  if G is not None: # no need to re-compute, just return it
    return G;

  if alpha is None:
    alpha = 0.5; 

  if G is None:
    G1 = compute_matrix_vec_grad(Y,params); 
    G2 = compute_matrix_vec_grad2(Y,params); 
    G = (1.0 - alpha)*G1 + alpha*G2; 

  if flag_save:
    extras.update({'G':G});

  return G; 

def compute_matrix_tensor_div2(Y,params,extras=None):
  """ Divergence acting on tensor fields with extra 
      averaging to help avoid the checkerboard instability. 
  """

  if extras is not None:
    D,G,flag_save = tuple(map(extras.get,['D','G','flag_save']));
  else:
    D,G = (None,None);
    flag_save = None; 

  if flag_save is None:
    flag_save = False; 

  if D is not None: # no need to re-compute, just return it
    return D;

  if G is None:
    G = compute_matrix_tensor_grad2(Y,params,extras);

  D = -G.T; 

  if flag_save:
    extras.update({'D':D});

  return D;

def compute_matrix_vec_div2(Y,params,extras=None):
  """ Divergence acting on vector fields with extra 
      averaging to help avoid the checkerboard instability. 
  """

  if extras is not None:
    D,G,flag_save = tuple(map(extras.get,['D','G','flag_save']));
  else:
    D,G = (None,None);
    flag_save = None; 

  if flag_save is None:
    flag_save = False; 

  if D is not None: # no need to re-compute, just return it
    return D;

  if G is None:
    G = compute_matrix_vec_grad2(Y,params,extras);

  D = -G.T; 

  if flag_save:
    extras.update({'D':D});

  return D;

def compute_matrix_tensor_div3(Y,params,extras=None):
  """ Divergence acting on tensor fields with extra 
      averaging to help avoid the checkerboard instability. 
  """

  if extras is not None:
    D,G,flag_save = tuple(map(extras.get,['D','G','flag_save']));
  else:
    D,G = (None,None);
    flag_save = None; 

  if flag_save is None:
    flag_save = False; 

  if D is not None: # no need to re-compute, just return it
    return D;

  if G is None:
    G = compute_matrix_tensor_grad3(Y,params,extras);

  D = -G.T; 

  if flag_save:
    extras.update({'D':D});

  return D;

def compute_matrix_vec_div3(Y,params,extras=None):
  """ Divergence acting on vector fields with extra 
      averaging to help avoid the checkerboard instability. 
  """

  if extras is not None:
    D,G,flag_save = tuple(map(extras.get,['D','G','flag_save']));
  else:
    D,G = (None,None);
    flag_save = None; 

  if flag_save is None:
    flag_save = False; 

  if D is not None: # no need to re-compute, just return it
    return D;

  if G is None:
    G = compute_matrix_vec_grad3(Y,params,extras);

  D = -G.T; 

  if flag_save:
    extras.update({'D':D});

  return D;

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

  #if flag_save: 
  #  extras.update({'matrix_Gamma_op':matrix_Gamma_op});

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

def compute_K_visco_grad(Y,params,extras=None):

  # get params data 
  num_mesh_x = params['num_mesh_x']; num_mesh_y = params['num_mesh_y'];
  num_mesh_pts = num_mesh_x*num_mesh_y;
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaX = params['deltaX']; deltaV = deltaX_sq = deltaX*deltaX;
  mu = params['mu']; Y_I = params['Y_I'];
  flag_incompressible = params['flag_incompressible'];

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);

  if extras is not None:
    G,matrix_wp = tuple(map(extras.get,['G','matrix_wp']));
  else:
    G,matrix_wp = (None,None);

  if G is None:
    G = compute_matrix_tensor_grad(Y,params);

  if matrix_wp is None:
    matrix_wp = compute_matrix_wp(Y,params);

  if flag_incompressible:
    G = np.matmul(G,matrix_wp.T); # modify to \nabla \wp^T 

  # local transpose the individual blocks of the tensor
  # split into tensor with indexing (mesh_I,tensor_i,tensor_j,mesh_J,vec_i)
  GG = G.reshape(num_mesh_pts,num_dim,num_dim,num_mesh_pts,num_dim); 
  theta = np.expand_dims(fluid_theta,(1,2,3,4)); # expand for broadcasting
  K_GG = mu*theta*GG/deltaV; # scaling 1/deltaV for density
  K_GG_flip = K_GG.transpose((0,2,1,3,4));
  K_visco_grad = (K_GG + K_GG_flip).reshape(num_mesh_pts*num_dim_sq,num_mesh_pts*num_dim);

  return K_visco_grad;

def compute_M_sym(Y,params):

  num_mesh_x = params['num_mesh_x']; num_mesh_y = params['num_mesh_y'];
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaX = params['deltaX']; mu = params['mu']; Y_I = params['Y_I'];

  # construct linear operator 
  # M_sym*A = A + A^T 
  M_sym = np.zeros((num_dim_sq,num_dim_sq));
  for a1 in range(0,num_dim):
    for a2 in range(0,num_dim):
      i1 = a1*num_dim + a2; j1 = i1; j2 = a2*num_dim + a1;
      M_sym[i1,j1] += 1.0; M_sym[i1,j2] += 1.0;

  return M_sym; 

def compute_M_sym_factor(Y,params,extras=None):
 
  if extras is not None:
    M_sym,R_sym,flag_save = tuple(map(extras.get,['M_sym','R_sym','flag_save']));
    if flag_save is None:
      flag_save = False;  
  else:
    M_sym = None; R_sym = None;
    flag_save = False; 
   
  if R_sym is not None:
    return R_sym; # no need to recompute, just return  

  num_mesh_x = params['num_mesh_x']; num_mesh_y = params['num_mesh_y'];
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaX = params['deltaX']; mu = params['mu']; Y_I = params['Y_I'];

  if M_sym is None:
    M_sym = compute_M_sym(Y,params);
           
  # compute using svd the factor, M_sym = RR^T
  U,S,Vh = np.linalg.svd(M_sym,hermitian=True);
  sqrt_S = np.diag(np.sqrt(S));
  R_sym = np.matmul(U,sqrt_S);

  # save calculated results 
  if flag_save:
    extras['M_sym'] = M_sym; 
    extras['R_sym'] = R_sym; 

  return R_sym; 

def compute_K_visco_factor(Y,params,extras=None):

  # get params data 
  num_mesh_x = params['num_mesh_x']; num_mesh_y = params['num_mesh_y'];
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaX = params['deltaX']; mu = params['mu']; Y_I = params['Y_I'];
  deltaV = deltaX_sq = deltaX*deltaX;

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  K_visco_factor = np.zeros((num_mesh_x*num_mesh_y*num_dim_sq,num_mesh_x*num_mesh_y*num_dim_sq)); 

  R_sym = compute_M_sym_factor(Y,params,extras);

  # use the block structure to compute the factors
  vec_I = np.zeros(2,dtype=int);
  for j in range(0,num_mesh_y): 
    for i in range(0,num_mesh_x):
      vec_I[0] = i; vec_I[1] = j; 
      vec_Im1 = vec_I + 0; vec_Ip1 = vec_I + 0; # make copies
      I0_mesh = vec_I[1]*num_mesh_x*num_dim_sq + vec_I[0]*num_dim_sq + 0;
      I0_theta = vec_I[1]*num_mesh_x + vec_I[0] + 0;

      for a in range(0,num_dim):
        for b in range(0,num_dim):
          i1 = I0_mesh; i2 = i1 + num_dim_sq;
          sqrt_val = np.sqrt(mu*fluid_theta[I0_theta]/deltaV); # scaling 1/deltaV for density;
          K_visco_factor[I0_mesh + a*num_dim + b,i1:i2] = sqrt_val*R_sym[a*num_dim + b,:]; 

  return K_visco_factor;


def compute_dot_F(Y,params,extras=None):
 
  if extras is not None:
    G, = tuple(map(extras.get,['G']));
  else:
    G = None;

  if G is None:
    G = compute_matrix_tensor_grad(Y,params);
 
  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,
    ['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 

  mu,Y_I,rho = tuple(map(params.get,
    ['mu','Y_I','rho']));

  fluid_phi = get_comp(Y,'I1_fluid_phi','I2_fluid_phi',Y_I);
  fluid_p = get_comp(Y,'I1_fluid_p','I2_fluid_p',Y_I);
  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  num_fluid_theta = fluid_theta.shape[0];

  dot_F = np.matmul(G,fluid_p/rho);

  return dot_F;


def compute_K_visco_dot_F(Y,params,extras=None):

  if extras is not None:
    dot_F,G = tuple(map(extras.get,['dot_F','G']));
  else:
    dot_F,G = (None,None);

  if dot_F is None:
    dot_F = compute_dot_F(Y,params,{'G':G});

  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,
    ['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;

  m,rho,gamma,mu,Y_I,c_v,c_v_I = tuple(map(params.get,
    ['m','rho','gamma','mu','Y_I','c_v','c_v_I']));

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  
  # local transpose the individual blocks of the tensor
  dot_FF = dot_F.reshape(num_mesh_x*num_mesh_y,num_dim,num_dim);  
  theta = np.expand_dims(fluid_theta,(1,2)); # expand for broadcasting
  K_dFFF = mu*theta*dot_FF/deltaV; # density so scaling by 1/deltaV
  K_dFFF_flip = K_dFFF.transpose((0,2,1));
  K_visco_dot_F = (K_dFFF + K_dFFF_flip).flatten();

  return K_visco_dot_F; 

def compute_partial_theta_K_visco_dot_F(Y,params,extras=None):

  if extras is not None:
    dot_F,G = tuple(map(extras.get,['dot_F','G']));
  else:
    dot_F,G = (None,None);

  if dot_F is None:
    dot_F = compute_dot_F(Y,params,{'G':G});

  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,
    ['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 

  m,rho,gamma,mu,Y_I,c_v,c_v_I,deltaX = tuple(map(params.get,
    ['m','rho','gamma','mu','Y_I','c_v','c_v_I','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX;

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  
  # local transpose the individual blocks of the tensor
  dot_FF = dot_F.reshape(num_mesh_x*num_mesh_y,num_dim,num_dim);  
  theta = np.expand_dims(fluid_theta,(1,2)); # expand for broadcasting
  K_dFFF = mu*dot_FF/deltaV; # scaling by 1/deltaV for density
  K_dFFF_flip = K_dFFF.transpose((0,2,1));
  partial_theta_K_visco_dot_F = (K_dFFF + K_dFFF_flip).flatten();

  return partial_theta_K_visco_dot_F; 


def compute_K_heat(Y,params,extras=None):
  """ We assume $\tilde{\kappa} = \tilde{\kappa}_0 \theta^2$
    then the K_{heat} block is $\tilde{\kappa} I = \tilde{\kappa}_0 \theta^2$.
    This yields  
    $\bar{K}_{\theta,\theta} = -\mbox{\small div}(\tilde{\kappa}\nabla} 
    = -\mbox{\small div}(\tilde{\kappa}_0 \theta^2\nabla$.
  """
    
  if extras is not None:
    G,D = tuple(map(extras.get,['G','D']));
  else:
    G,D = (None,None);

  if G is None:
    G = compute_matrix_vec_grad(Y,params);

  if D is None:
    D = compute_matrix_vec_div(Y,params);

  # get params data 
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,
    ['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  num_mesh_pts = num_mesh_x*num_mesh_y;

  m,rho,gamma,mu,Y_I,c_v,c_v_I,kappa_0 = tuple(map(params.get,
    ['m','rho','gamma','mu','Y_I','c_v','c_v_I','kappa_0']));

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  num_fluid_theta = fluid_theta.shape[0];

  # if creating new array each time, then faster to use
  theta = np.expand_dims(fluid_theta,(1,2)); # for broadcasting
  GG = G.reshape(num_mesh_pts,num_dim,num_mesh_pts);
  A1 = kappa_0*np.power(theta,2)*GG;
  A2 = A1.reshape(num_mesh_pts*num_dim,num_mesh_pts);
  K_heat = np.matmul(-D,A2);  

  # if array is pre-created and re-used then this is probably faster 
  #ii1 = range(0,num_fluid_theta);
  #K_heat[ii1,ii1] = kappa_0*np.power(fluid_theta,2);
  
  return K_heat; 

def compute_K_heat2(Y,params,extras=None):
  """ K_heat based on a finite-volume-like model using 
  local transfers similar to Fourier's Law formulated in
  terms of 1/theta.  Gives final action similar to 
  central difference Laplacian in terms of temperature.
  Operator itself operates on 1/theta and is positive 
  semi-definite.  """
    
  # get params data
  kappa_0,num_mesh_x,num_mesh_y,num_dim,c_v,c_v_I,deltaX,mu,Y_I = tuple(map(params.get,
    ['kappa_0','num_mesh_x','num_mesh_y','num_dim','c_v','c_v_I','deltaX','mu','Y_I']));
  num_mesh_pts = num_mesh_x*num_mesh_y;
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;

  ii = c_v_I; 
  c_F = c_v[ii['fluid']];

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  num_fluid_theta = fluid_theta.shape[0];

  K_heat = np.zeros((num_fluid_theta,num_fluid_theta));
  x1 = np.linspace(0,num_mesh_x-1,num_mesh_x); 
  x2 = np.linspace(0,num_mesh_y-1,num_mesh_y);
  I = np.meshgrid(x1,x2); 
  I1 = np.rint(I[0].flatten()).astype(dtype=int);  
  I2 = np.rint(I[1].flatten()).astype(dtype=int);  
  II = np.array([I1,I2]).T; 

  Iip1 = np.zeros(II.shape,dtype=int); Iip1[:,:] = II; Iip1[:,0] = II[:,0] + 1; # periodic boundaries 
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
  
  theta_I0 = fluid_theta[II0];
  theta_Iip1 = fluid_theta[IIip1];
  theta_Iim1 = fluid_theta[IIim1];
  theta_Ijp1 = fluid_theta[IIjp1];
  theta_Ijm1 = fluid_theta[IIjm1];

  c = kappa_0/(c_F*c_F*deltaV);
  K_heat[II0,II0] = c*theta_I0*(theta_Iip1 + theta_Iim1 + theta_Ijp1 + theta_Ijm1);
  K_heat[II0,IIip1] = -c*theta_I0*theta_Iip1;
  K_heat[II0,IIim1] = -c*theta_I0*theta_Iim1;
  K_heat[II0,IIjp1] = -c*theta_I0*theta_Ijp1;
  K_heat[II0,IIjm1] = -c*theta_I0*theta_Ijm1;

  #K_heat = K_heat/deltaV; # scaling 1/deltaV for density 
 
  return K_heat; 

def compute_R_visco(Y,params,extras=None):
  """ R_visco factor so that R_visco*R_visco^T = K_visco
  used for computing thermal fluctuations.
  """ 
  m,rho,gamma,mu,kappa_0,Y_I = tuple(map(params.get,['m','rho','gamma','mu','kappa_0','Y_I']));
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX;
   
  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);

  if extras is not None:
    flag_save,matrix_tensor_div,dot_F = tuple(map(extras.get,['flag_save','matrix_tensor_div','dot_F']));
  else:
    flag_save,matrix_tensor_div,dot_F = (None,None,None);
  
  if flag_save is None:
    flag_save = False;

  if matrix_tensor_div is None:
    matrix_tensor_div = compute_matrix_tensor_div(Y,params,extras_matrix_tensor_div); 

  if dot_F is None:
    extras_K_visco_dot_F = extras['extras_K_visco_dot_F'];
    dot_F = compute_dot_F(Y,params,extras_K_visco_dot_F);

  R_visco = np.zeros((num_mesh_x*num_mesh_y*num_dim_sq,
                      num_mesh_x*num_mesh_y*num_dim_sq));
  for i1 in range(0,num_mesh_x):
    for j1 in range(0,num_mesh_y):
      I0_mesh = j1*num_mesh_x*num_dim_sq + i1*num_dim_sq;
      I_theta = j1*num_mesh_x + i1;
      sqrt_half_mu_theta = np.sqrt(0.5*mu*fluid_theta[I_theta]);
      for a in range(0,num_dim):
        for b in range(0,num_dim):
          R_visco[I0_mesh + b*num_dim + a,I0_mesh + b*num_dim + a] += sqrt_half_mu_theta;
          R_visco[I0_mesh + b*num_dim + a,I0_mesh + a*num_dim + b] += sqrt_half_mu_theta;
  
  R_visco = R_visco/np.sqrt(deltaV); # scaling by (1/deltaV)^{1/2} needed for density
 
  if extras is not None and 'dot_F' in extras: # warning: invalidate, so do not forget to update each time 
    extras.update({'dot_F':None}); 

  if flag_save: 
    extras.update({'matrix_tensor_div':matrix_tensor_div}); 
    
  return R_visco; 

def compute_R_heat(Y,params,extras):
  m,rho,gamma,mu,kappa_0,Y_I = tuple(map(params.get,['m','rho','gamma','mu','kappa_0','Y_I']));
  num_mesh_x,num_mesh_y,num_dim,deltaX = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim','deltaX']));
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX;
   
  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);

  # note, may need matrix_tensor_div to be for vectors and not tensors
  # requiring some down-selection of entries  

  if extras is not None:
    flag_save,matrix_vec_div,matrix_tensor_div = tuple(map(extras.get,['flag_save','matrix_vec_div','matrix_tensor_div']));
  else:
    flag_save,matrix_vec_div,matrix_tensor_div = (None,None,None);
 
  if flag_save is None:
    flag_save = False;

  if matrix_vec_div is None: 
    if matrix_tensor_div is None:
      matrix_tensor_div = compute_matrix_tensor_div(Y,params,extras_matrix_tensor_div);
    matrix_vec_div = extract_matrix_vec_div(matrix_tensor_div,params);

  sqrt_tilde_kappa = np.sqrt(kappa_0*fluid_theta*fluid_theta/deltaV); # kappa_0*theta^2/dV 
  n1 = num_mesh_x*num_mesh_y*num_dim; n2 = num_mesh_x*num_mesh_y*num_dim;
  sqrt_tilde_kappa_I = np.zeros((n1,n2));
  k1 = range(0,n1,num_dim); k2 = range(0,n1,num_dim);
  sqrt_tilde_kappa_I[k1,k2] = sqrt_tilde_kappa;
  k1 = range(1,n1,num_dim); k2 = range(1,n1,num_dim);
  sqrt_tilde_kappa_I[k1,k2] = sqrt_tilde_kappa;
  R_heat = -np.dot(matrix_vec_div,sqrt_tilde_kappa_I);

  #R_heat = R_heat/np.sqrt(deltaV);

  if flag_save:
    extras.update({'matrix_vec_div':matrix_vec_div});
 
  return R_heat; 


def compute_R_heat2(Y,params,extras):

  """ 
    Compute the factor $K_heat = RR^T$.

    This uses the discrete operator based on finite-volume-like method
    and boxes.   This yields the factorization of the form 
  
    R = -D, where 
    $[G (1/\theta)]_{(I_1 + I_2)/2} = s_{I_1,I_2}\sqrt{\theta_{I_1}\theta_{I_2}}\left(1/\theta_{I_2} - 1/\theta_{I_1}
    \right)$, where $s_{I_1,I_2} = -1$ is $I_1 < I_2$ and $+1$ otherwise.  We claim
     that in fact we have $K_{heat} = -D\cdot G$ gives the operator. 

  """
  # get params data
  kappa_0,num_mesh_x,num_mesh_y,num_dim,c_v,c_v_I,deltaX,mu,Y_I = tuple(map(params.get,
    ['kappa_0','num_mesh_x','num_mesh_y','num_dim','c_v','c_v_I','deltaX','mu','Y_I']));
  num_mesh_pts = num_mesh_x*num_mesh_y;
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;

  ii = c_v_I; 
  c_F = c_v[ii['fluid']];

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  num_fluid_theta = fluid_theta.shape[0];

  wG = np.zeros((num_fluid_theta,num_dim,num_fluid_theta)); # weighted gradient
  x1 = np.linspace(0,num_mesh_x-1,num_mesh_x); 
  x2 = np.linspace(0,num_mesh_y-1,num_mesh_y);
  I = np.meshgrid(x1,x2); 
  I1 = np.rint(I[0].flatten()).astype(dtype=int);  
  I2 = np.rint(I[1].flatten()).astype(dtype=int);  
  II = np.array([I1,I2]).T; 

  Iip1 = np.zeros(II.shape,dtype=int); Iip1[:,:] = II; Iip1[:,0] = II[:,0] + 1;  # periodic boundaries
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
  
  theta_I0 = fluid_theta[II0];
  theta_Iip1 = fluid_theta[IIip1];
  theta_Iim1 = fluid_theta[IIim1];
  theta_Ijp1 = fluid_theta[IIjp1];
  theta_Ijm1 = fluid_theta[IIjm1];

  # we use staggered convention that I0^(d) is always 
  # to the left or below the cell-center I0^{d).
  # we construct the weighted gradient operator (wG) 
  c = np.sqrt(kappa_0/(c_F*c_F*deltaV));
  wG[II0,0,II0] = c*np.sqrt(theta_I0*theta_Iim1);
  wG[II0,0,IIim1] = -c*np.sqrt(theta_I0*theta_Iim1);
  wG[II0,1,II0] = c*np.sqrt(theta_I0*theta_Ijm1);
  wG[II0,1,IIjm1] = -c*np.sqrt(theta_I0*theta_Ijm1);

  wwG = wG.reshape(II0.shape[0]*num_dim,II0.shape[0]);

  R_heat = wwG.T;  # @@@ double-check transpose here 
  
  return R_heat; 


def extract_matrix_vec_div(matrix_tensor_div,params):
    num_mesh_x,num_mesh_y,num_dim = tuple(map(params.get,['num_mesh_x','num_mesh_y','num_dim']));
    num_dim_sq = num_dim*num_dim;
    matrix_vec_div = np.zeros((num_mesh_x*num_mesh_y,num_mesh_x*num_mesh_y*num_dim)); 
    ii1 = 0; ii2 = ii1 + num_mesh_x*num_mesh_y;
    jj1 = 0; jj2 = jj1 + num_mesh_x*num_mesh_y*num_dim;
    i1 = 0; i2 = i1 + num_mesh_x*num_mesh_y*num_dim;
    j1 = 0; j2 = j1 + num_mesh_x*num_mesh_y*num_dim_sq;
    matrix_vec_div[ii1:ii2,jj1:jj2:num_dim] = matrix_tensor_div[i1:i2:num_dim,j1:j2:num_dim_sq];
    ii1 = 0; ii2 = ii1 + num_mesh_x*num_mesh_y;
    jj1 = 1; jj2 = jj1 + num_mesh_x*num_mesh_y*num_dim;
    i1 = 0; i2 = i1 + num_mesh_x*num_mesh_y*num_dim;
    j1 = 1; j2 = j1 + num_mesh_x*num_mesh_y*num_dim_sq;
    matrix_vec_div[ii1:ii2,jj1:jj2:num_dim] = matrix_tensor_div[i1:i2:num_dim,j1:j2:num_dim_sq];

    return matrix_vec_div; 

def compute_partial_theta_K_heat(Y,params,extras=None):
  """ We assume $\tilde{\kappa} = \tilde{\kappa}_0 \theta^2$
    then the K_{heat} block is $\tilde{\kappa} I = \tilde{\kappa}_0 \theta^2$.
    This yields  
    $\bar{K}_{\theta,\theta} = -\mbox{\small div}(\tilde{\kappa}\nabla} 
    = -\mbox{\small div}(\tilde{\kappa}_0 \theta^2\nabla$ and 
    $\frac{\partial \bar{K}_{\theta,\theta}}{\partial \theta}
    = -\mbox{\small div}(2\tilde{\kappa}_0 \theta \nabla
    $ 
  """ 
  # get params data
  kappa_0 = params['kappa_0']; 
  num_mesh_x = params['num_mesh_x']; num_mesh_y = params['num_mesh_y'];
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaX = params['deltaX']; mu = params['mu']; Y_I = params['Y_I'];

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  num_fluid_theta = fluid_theta.shape[0];

  if extras is not None:
    flag_save,matrix_vec_div,matrix_tensor_div,matrix_scalar_grad = tuple(map(
      extras.get,['flag_save','matrix_vec_div','matrix_tensor_div','matrix_scalar_grad']));
  else:
    flag_save,matrix_vec_div,matrix_tensor_div,matrix_scalar_grad = (None,None,None,None);

  if matrix_vec_div is None: 
    matrix_vec_div = extract_matrix_vec_div(matrix_tensor_div,params);

  if matrix_scalar_grad is None:
    matrix_scalar_grad = np.transpose(matrix_vec_div);

  partial_tilde_kappa = 2.0*kappa_0*fluid_theta;
  partial_theta_tilde_kappa_grad = np.zeros(matrix_scalar_grad.shape);
  for d in range(0,num_dim):
    i1 = d; i2 = i1 + num_fluid_theta*num_dim;
    j1 = 0; j2 = j1 + num_fluid_theta;
    ii1 = d; ii2 = ii1 + num_fluid_theta*num_dim;
    jj1 = 0; jj2 = jj1 + num_fluid_theta;
    partial_theta_tilde_kappa_grad[i1:i2:num_dim,j1:j2] = partial_tilde_kappa*matrix_scalar_grad[ii1:ii2:num_dim,jj1:jj2];

  # @@@ double check how to reduce to scalar 
  partial_theta_K_heat = np.sum(np.dot(matrix_vec_div,partial_theta_tilde_kappa_grad),1)/deltaV; # scaling to match K_heat above (double-check)
  
  return partial_theta_K_heat; 

def compute_partial_theta_K_heat2(Y,params,extras=None):
  """   
    We use that this can be computed for K_heat2 the same 
    as multiplying by 1/\theta, since it removes a factor
    from the product in each line.  We could optimize by constructing
    just this special form and sum. 


  """ 
  
  # get params data
  kappa_0,num_mesh_x,num_mesh_y,num_dim,c_v,c_v_I,deltaX,mu,Y_I = tuple(map(params.get,
    ['kappa_0','num_mesh_x','num_mesh_y','num_dim','c_v','c_v_I','deltaX','mu','Y_I']));
  num_mesh_pts = num_mesh_x*num_mesh_y;
  num_dim_sq = num_dim*num_dim; # tensor dimension 
  deltaV = deltaX_sq = deltaX*deltaX;

  ii = c_v_I; 
  c_F = c_v[ii['fluid']];

  fluid_theta = get_comp(Y,'I1_fluid_theta','I2_fluid_theta',Y_I);
  num_fluid_theta = fluid_theta.shape[0];

  partial_theta_K_heat = np.zeros(num_fluid_theta);
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
  
  theta_I0 = fluid_theta[II0];
  theta_Iip1 = fluid_theta[IIip1];
  theta_Iim1 = fluid_theta[IIim1];
  theta_Ijp1 = fluid_theta[IIjp1];
  theta_Ijm1 = fluid_theta[IIjm1];

  c = kappa_0/(c_F*c_F*deltaX_sq); # @@@ double-check deltaX_sq
  partial_theta_K_heat[II0] = c*(theta_Iip1 + theta_Iim1 + theta_Ijp1 + theta_Ijm1);
  partial_theta_K_heat[II0] += -c*theta_I0;
  partial_theta_K_heat[II0] += -c*theta_I0;
  partial_theta_K_heat[II0] += -c*theta_I0;
  partial_theta_K_heat[II0] += -c*theta_I0;

  return partial_theta_K_heat; 

def compute_partial_p_dot_F(Y,params,extras=None):

  rho = params['rho'];

  Y_parts = dd = get_parts(Y,params);
  fluid_phi,fluid_p,fluid_theta = tuple(map(dd.get,['fluid_phi','fluid_p','fluid_theta']));

  num_fluid_phi = fluid_phi.shape[0]; num_fluid_p = fluid_p.shape[0]; num_fluid_theta = fluid_theta.shape[0];

  partial_p_dot_F = np.zeros(num_fluid_p)

  if extras is not None:
    matrix_tensor_div, = tuple(map(params.get,['compute_matrix_tensor_div']));
  else:
    matrix_tensor_div = None; 

  if matrix_tensor_div is None:
    matrix_tensor_div = compute_matrix_tensor_div(Y,params);

  # WARNING: Need to get correct matrix_tensor_div (vector version I think)
  matrix_tensor_grad = np.transpose(matrix_tensor_div); 

  partial_p_dot_F = matrix_tensor_grad/rho;
  
  return partial_p_dot_F;

def compute_M_S_j(Y,params):
  pass;

def compute_K0_j(Y,params):
  pass;

def compute_L0(Y,params):
  pass; 

def compute_bar_L(Y,params,extras=None):
  m,rho,gamma,mu,Y_I,c_v,c_v_I,deltaX = tuple(map(
    params.get,['m','rho','gamma','mu','Y_I','c_v','c_v_I','deltaX']));
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim;
  deltaV = deltaX_sq = deltaX*deltaX;

  get_parts = params['get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_p,particle_theta = tuple(map(
    dd.get,['particle_q','particle_p','particle_theta']));
  fluid_phi,fluid_p,fluid_theta = tuple(map(
    dd.get,['fluid_phi','fluid_p','fluid_theta']));
  interface_q,interface_p,interface_theta = tuple(map(
    dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_p,I2_particle_p = tuple(map(
    dd.get,['I1_particle_p','I2_particle_p']));
  I1_particle_theta,I2_particle_theta = tuple(map(
    dd.get,['I1_particle_theta','I2_particle_theta']));

  I1_fluid_phi,I2_fluid_phi = tuple(map(
    dd.get,['I1_fluid_phi','I2_fluid_phi']));
  I1_fluid_p,I2_fluid_p = tuple(map(
    dd.get,['I1_fluid_p','I2_fluid_p']));
  I1_fluid_theta,I2_fluid_theta = tuple(map(
    dd.get,['I1_fluid_theta','I2_fluid_theta']));

  I1_interface_q,I2_interface_q = tuple(map(    
    dd.get,['I1_interface_q','I2_interface_q']));
  I1_interface_p,I2_interface_p = tuple(map(
    dd.get,['I1_interface_p','I2_interface_p']));
  I1_interface_theta,I2_interface_theta = tuple(map(
    dd.get,['I1_interface_theta','I2_interface_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_p = particle_p.shape[0];
  num_particle_theta = particle_theta.shape[0]; 
  num_fluid_phi = fluid_phi.shape[0]; num_fluid_p = fluid_p.shape[0]; 
  num_fluid_theta = fluid_theta.shape[0]; 
  num_interface_q = interface_q.shape[0]; 
  num_interface_p = interface_p.shape[0]; 
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

  num_q = num_particle_q;
  i1 = I1_particle_q; i2 = I2_particle_q; ii = range(i1,i2);
  j1 = I1_particle_p; j2 = I2_particle_p; jj = range(j1,j2);
  bar_L[ii,jj] = 1.0;  # identity tensor 
  bar_L[jj,ii] = -1.0; # negative identity tensor

  num_q = num_fluid_phi;
  i1 = I1_fluid_phi; i2 = I2_fluid_phi; ii = range(i1,i2);
  j1 = I1_fluid_p; j2 = I2_fluid_p; jj = range(j1,j2);
  bar_L[ii,jj] = 1.0/deltaV;  # identity tensor (energy density) 
  bar_L[jj,ii] = -1.0/deltaV; # negative identity tensor (energy density)

  num_q = num_interface_q;
  i1 = I1_interface_q; i2 = I2_interface_q; ii = range(i1,i2);
  j1 = I1_interface_p; j2 = I2_interface_p; jj = range(j1,j2);
  bar_L[ii,jj] = 1.0;  # identity tensor 
  bar_L[jj,ii] = -1.0; # negative identity tensor

  ll = c_v_I; 

  # particle entropy contributions partial_q S
  i1 = I1_particle_p; i2 = I2_particle_p;
  j1 = I1_particle_theta; j2 = I2_particle_theta;
  k1 = 0; k2 = k1 + num_particle_q;
  partial_theta_S_j = particle_theta*c_v[ll['particle']];
  bar_L[i1:i2,j1:j2] = np.expand_dims(D_S_j[ll['particle']][k1:k2]/partial_theta_S_j,1);
  bar_L[j1:j2,i1:i2] = -np.transpose(bar_L[i1:i2,j1:j2]);

  # fluid entropy contributions partial_q S
  i1 = I1_fluid_p; i2 = I2_fluid_p;
  j1 = I1_fluid_theta; j2 = I2_fluid_theta;
  k1 = 0; k2 = k1 + num_fluid_phi;
  partial_theta_S_j = fluid_theta*c_v[ll['fluid']];
  for d in range(0,num_dim):
    ii = range(i1 + d,i2,num_dim); jj = range(j1,j2);
    bar_L[ii,jj] = D_S_j[ll['fluid']][k1 + d:k2:num_dim]/(partial_theta_S_j*deltaV); # deltaV for entropy density
    bar_L[jj,ii] = -np.transpose(bar_L[ii,jj]);

  # interface entropy contributions partial_q S
  i1 = I1_interface_p; i2 = I2_interface_p;
  j1 = I1_interface_theta; j2 = I2_interface_theta;
  k1 = 0; k2 = k1 + num_interface_q;
  partial_theta_S_j = interface_theta*c_v[ll['interface']];
  bar_L[i1:i2,j1:j2] = np.expand_dims(D_S_j[ll['interface']][k1:k2]/partial_theta_S_j,1);
  bar_L[j1:j2,i1:i2] = -np.transpose(bar_L[i1:i2,j1:j2]);

  # there is a separate entropy $S^{(j)}$ for each heat-body 
  I_in = {'i1':[0],'i2':[Y.shape[0]]};
  I_local_in = {'i1':[0],'i2':[Y.shape[0]]};
  I_local_out = {'i1':[0],'i2':[Y.shape[0]]};
  I_out = {'i1':[0],'i2':[Y.shape[0]]};

  bar_L_indices = {'I_in':I_in,'I_out':I_out,
                   'I_local_in':I_local_in,'I_local_out':I_local_out};

  return bar_L,bar_L_indices;  

def get_bar_K_j_I(params):
  bar_K_j_I_list = {'particle':0,'fluid':1,'interface':2};
  return bar_K_j_I_list;

def compute_bar_K_j(Y,params,extras=None):
  m,rho,gamma,mu,Y_I,c_v,c_v_I = tuple(map(
    params.get,['m','rho','gamma','mu','Y_I','c_v','c_v_I']));
  num_dim = params['num_dim'];
  flag_incompressible, = tuple(map(params.get,['flag_incompressible']));
  num_dim_sq = num_dim*num_dim;

  num_mesh_x,num_mesh_y,deltaX = \
    tuple(map(params.get,['num_mesh_x','num_mesh_y','deltaX']));
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  deltaV = deltaX_sq = deltaX*deltaX; 

  kappa_P_I,kappa_F_I = tuple(map(params.get,['kappa_P_I','kappa_F_I']));

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_p,particle_theta = tuple(map(
    dd.get,['particle_q','particle_p','particle_theta']));
  fluid_phi,fluid_p,fluid_theta = tuple(map(
    dd.get,['fluid_phi','fluid_p','fluid_theta']));
  interface_q,interface_p,interface_theta = tuple(map(
    dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_p,I2_particle_p = tuple(map(
    dd.get,['I1_particle_p','I2_particle_p']));
  I1_particle_theta,I2_particle_theta = tuple(map(
    dd.get,['I1_particle_theta','I2_particle_theta']));

  I1_fluid_phi,I2_fluid_phi = tuple(map(
    dd.get,['I1_fluid_phi','I2_fluid_phi']));
  I1_fluid_p,I2_fluid_p = tuple(map(
    dd.get,['I1_fluid_p','I2_fluid_p']));
  I1_fluid_theta,I2_fluid_theta = tuple(map(
    dd.get,['I1_fluid_theta','I2_fluid_theta']));

  I1_interface_q,I2_interface_q = tuple(map(    
    dd.get,['I1_interface_q','I2_interface_q']));
  I1_interface_p,I2_interface_p = tuple(map(
    dd.get,['I1_interface_p','I2_interface_p']));
  I1_interface_theta,I2_interface_theta = tuple(map(
    dd.get,['I1_interface_theta','I2_interface_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_p = particle_p.shape[0];
  num_particles = num_particle_q//num_dim; 
  num_particle_theta = particle_theta.shape[0]; 
  num_fluid_phi = fluid_phi.shape[0]; num_fluid_p = fluid_p.shape[0]; 
  num_fluid_theta = fluid_theta.shape[0]; 
  num_interface_q = interface_q.shape[0]; 
  num_interface_p = interface_p.shape[0]; 
  num_interface_theta = interface_theta.shape[0];

  # extras handling   
  if extras is not None:
    extras_matrix_tensor_div,extras_K_visco_grad = tuple(map(
      extras.get,['extras_matrix_tensor_div','extras_K_visco_grad'])); 
    extras_K_visco_dot_F,extras_K_heat = tuple(map(
      extras.get,['extras_K_visco_dot_F','extras_K_heat']));
    extras_matrix_wp, = tuple(map(
      extras.get,['extras_matrix_wp']));
    flag_save,flag_save_energy_flux,flag_flux_check = tuple(map(
      extras.get,['flag_save','flag_save_energy_flux','flag_flux_check']));
  else:
    extras_matrix_tensor_div,extras_K_visco_grad = (None,None);
    extras_K_visco_dot_F,extras_K_heat = (None,None);
    extras_matrix_wp = None; 
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
  if flag_save:
    D_j_list = []; 
  I_in_list = []; I_out_list = []; 
  I_local_in_list = []; I_local_out_list = []; 
  if flag_save_energy_flux:
    energy_flux_list = [];  # one for each irreversible process 

  # particle case 
  j = c_v_I['particle'];  # j = 0
  if num_particle_theta > 1:
    ss = ""; 
    ss += "Expected num_particle_theta = 1, but num_particle_theta = " + str(num_particle_theta);
    ss += "Need a separate K_j for each particle, currently not implemented.";
    raise Exception(ss);
  I1_q = 0; I2_q = I1_q + num_particle_q; 
  I1_p = I2_q; I2_p = I1_p + num_particle_p;
  I1_theta = I2_p; I2_theta = I2_p + num_particle_theta;

  I_in = {'i1':[I1_particle_q,I1_particle_p,I1_particle_theta],
          'i2':[I2_particle_q,I2_particle_p,I2_particle_theta]};
  I_local_in = {'i1':[I1_q,I1_p,I1_theta],'i2':[I2_q,I2_p,I2_theta]};  
  I_local_out = {'i1':[I1_q,I1_p,I1_theta],'i2':[I2_q,I2_p,I2_theta]};  
  I_out = {'i1':[I1_particle_q,I1_particle_p,I1_particle_theta],
           'i2':[I2_particle_q,I2_particle_p,I2_particle_theta]}; 

  num_q = num_particle_q; num_p = num_particle_p; num_theta = num_particle_theta;
  bar_K_j = np.zeros((num_q + num_p + 1, num_q + num_p + 1)); 
  D_j = compute_D_j_particle(Y,params); ii = c_v_I;
  if flag_save: D_j_list.append(D_j);
  c_P = partial_theta_j_u_j = c_v[ii['particle']];
  theta_P = theta_j = particle_theta; I_theta = I2_p;
  dot_q = particle_p/m;
  bar_K_j[I1_p:I2_p,I1_p:I2_p] = theta_j*D_j; 
  bar_K_j[I1_p:I2_p,I_theta] = -theta_j*np.dot(D_j,dot_q)/partial_theta_j_u_j; 
  bar_K_j[I_theta,I1_p:I2_p] = np.transpose(bar_K_j[I1_p:I2_p,I_theta]);
  #bar_K_j[j,I_theta,I_theta] = theta_j*np.dot(dot_q,np.dot(D_j,dot_q))/(partial_theta_j_u_j*partial_theta_j_u_j);
  bar_K_j[I_theta,I_theta] = np.dot(dot_q,-1.0*bar_K_j[I1_p:I2_p,I_theta])/partial_theta_j_u_j;
  bar_K_j_list.append(bar_K_j);
  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  if flag_save_energy_flux or flag_flux_check:
    rate_kinetic_particle= np.sum(bar_K_j[I1_p:I2_p,I_theta].T*(c_P/theta_P)*(particle_p/m));
    rate_heat_particle = np.sum(bar_K_j[I_theta,I_theta]*(c_P/theta_P)*c_P);
    rate_total = rate_kinetic_particle + rate_heat_particle;

  if flag_save_energy_flux:
    energy_flux = {};
    energy_flux['rate_kinetic_particle'] = rate_kinetic_particle;
    energy_flux['rate_heat_particle'] = rate_heat_particle;
    energy_flux['rate_total'] = rate_total;
    energy_flux_list.append(energy_flux); 

  if flag_flux_check: 
    print("rate_kinetic_particle = %.2e"%rate_kinetic_particle);
    print("rate_heat_particle = %.2e"%rate_heat_particle);
    print("rate_total = %.2e"%(rate_total));

  # fluid case 
  # note subtle issues implementing correctly the operator, see $\square$ placement in notes.
  j = c_v_I['fluid'];  # j = 1
  if flag_save: D_j_list.append(None);
  I1_q = 0; I2_q = I1_q + num_fluid_phi; 
  I1_p = I2_q; I2_p = I1_p + num_fluid_p;
  I1_theta = I2_p; I2_theta = I2_p + num_fluid_theta;
  I_in = {'i1':[I1_fluid_phi,I1_fluid_p,I1_fluid_theta],
          'i2':[I2_fluid_phi,I2_fluid_p,I2_fluid_theta]};
  I_local_in = {'i1':[I1_q,I1_p,I1_theta],'i2':[I2_q,I2_p,I2_theta]};
  I_local_out = {'i1':[I1_q,I1_p,I1_theta],'i2':[I2_q,I2_p,I2_theta]};
  I_out = {'i1':[I1_fluid_phi,I1_fluid_p,I1_fluid_theta],
           'i2':[I2_fluid_phi,I2_fluid_p,I2_fluid_theta]};
  num_q = num_fluid_phi; num_p = num_fluid_p; num_theta = num_fluid_theta;
  bar_K_j = np.zeros((num_q + num_p + num_theta, num_q + num_p + num_theta)); 
  
  ii = c_v_I;
  c_F = partial_theta_j_u_j = c_v[ii['fluid']]; 
  theta_j = theta_F = fluid_theta;
  matrix_tensor_div = compute_matrix_tensor_div(Y,params,extras_matrix_tensor_div); 
  matrix_wp = compute_matrix_wp(Y,params,extras_matrix_wp); 
  K_visco_grad = compute_K_visco_grad(Y,params,extras_K_visco_grad);
  dot_F = compute_dot_F(Y,params);
  flag_debug = False;
  if flag_debug:
    debug_dir = params['debug_dir'];
    xx = get_mesh_xx(params);
    aa = dot_F.reshape(num_mesh_pts,num_dim_sq)
    filename = debug_dir + '/dot_F_01.vtr';
    fff=[];
    for d in range(0,num_dim_sq):
      ff = {};
      ff['field_name'] = 'dot_F_%.2d'%d;
      ff['NumberOfComponents'] = 1;
      ff['field_values'] = aa[:,d]
      fff.append(ff);
    write_vtr_data(filename,xx,fff); 

  if extras_K_visco_dot_F is not None:
    extras_K_visco_dot_F.update({'dot_F':dot_F});
  K_visco_dot_F = compute_K_visco_dot_F(Y,params,extras_K_visco_dot_F);
  K_heat = compute_K_heat2(Y,params,extras_K_heat);
  # compute the divergence of the gradient to obtain the upper-left block
  aa = -np.matmul(matrix_tensor_div,K_visco_grad); # -\nabla \cdot K_visco \nabla 
  if flag_incompressible: # grad projection already in K_visco_grad
    aa = np.matmul(matrix_wp,aa); # additional terms: -\wp \nabla \cdot K_visco \nabla \wp^T
  bar_K_j[I1_p:I2_p,I1_p:I2_p] = aa; 
  # Organize so each divergence tracked as if done separately for each x-location 
  # after we already multipled in the c_F*dx/theta_F(x) term.  Hence, $\square$ 
  # notation in our notes for this subtle ordering of operations in this linear operator. 
  # Expand first over the theta columns appropriately since vector quantities then apply 
  # the divergence operator, so would happen in the correct order (as in notes discussion).
  # The operator is given by action of  
  # $\frac{\mbox{\small div}\left(K_{visco} \square \dot{\mb{F}} \right)}{\partial_\tau u}$
  aa = K_visco_dot_F;  # need action of operator for multiplication by 1/theta_F(x_m) at each x-location 
  K_visco_square_dot_F = np.zeros((num_mesh_pts*num_dim_sq,num_mesh_pts)); # num_mesh_pts = num_theta 
  for d in range(0,num_dim_sq):
    ii1 = range(d,num_mesh_pts*num_dim_sq,num_dim_sq); jj1 = range(0,num_mesh_pts); 
    K_visco_square_dot_F[ii1,jj1] = aa[d::num_dim_sq]; # setting subset of entries

  div_K_visco_square_dot_F = np.matmul(matrix_tensor_div,K_visco_square_dot_F);
  if flag_incompressible: # multiply final result by $\wp$
    div_K_visco_square_dot_F = np.matmul(matrix_wp,div_K_visco_square_dot_F); # additional terms: \wp (--)
  bar_K_j[I1_p:I2_p,I1_theta:I2_theta] = div_K_visco_square_dot_F/partial_theta_j_u_j;

  # tranpose to get the other components of the tensor 
  bar_K_j[I1_theta:I2_theta,I1_p:I2_p] = np.transpose(bar_K_j[I1_p:I2_p,I1_theta:I2_theta]);
  # compute the lower-right block (dissipative work done that accumulates as internal energy tracked by temperature)
  # need to compute as if tensor dot product done at each x-location 
  ii2 = range(I1_theta,I2_theta); jj2 = range(I1_theta,I2_theta);
  for d in range(0,num_dim_sq):
#    bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta] += np.diag(dot_F[d::num_dim_sq]*K_visco_dot_F[d::num_dim_sq]);
    bar_K_j[ii2,jj2] += dot_F[d::num_dim_sq]*K_visco_dot_F[d::num_dim_sq];
  bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta] = bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta]/(partial_theta_j_u_j*partial_theta_j_u_j);
  bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta] += K_heat/(partial_theta_j_u_j*partial_theta_j_u_j);

  bar_K_j_list.append(bar_K_j);
  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  if flag_save_energy_flux or flag_flux_check:
    #rate_kinetic_fluid = 0;
    rate_kinetic_fluid = np.dot(np.dot(bar_K_j[I1_p:I2_p,I1_theta:I2_theta],(c_F*deltaV/theta_F)),(fluid_p/rho))*deltaV; 
    rate_heat_fluid = np.sum(np.dot(bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta],(c_F*deltaV/theta_F)))*c_F*deltaV;
    rate_total = rate_kinetic_fluid + rate_heat_fluid; 

  if flag_save_energy_flux:
    energy_flux = {};
    energy_flux['rate_kinetic_fluid'] = rate_kinetic_fluid; 
    energy_flux['rate_heat_fluid'] = rate_heat_fluid;
    energy_flux['rate_total'] = rate_total;
    energy_flux_list.append(energy_flux); 

  # check the energy exchanges (kinetic to heat energy)
  if flag_flux_check:
    print("");
    print("checking energy exchanges for fluid:"); 
    print("rate_kinetic_fluid = %.3e"%rate_kinetic_fluid);
    print("rate_heat_fluid = %.3e"%rate_heat_fluid); 
    print("rate_total = %.1e"%rate_total);
    print("");
 
  # interface case
  j = c_v_I['interface'];  # j = 2
  if num_interface_theta > 1:
    ss = ""; ss += "Expected num_interface_theta = 1, but num_interface_theta = " + str(num_interface_theta);
    ss += "Need a separate K_j for each interface, currently not implemented.";
    raise Exception(ss);
  num_q = num_particle_q + num_fluid_phi; num_p = num_particle_p + num_fluid_p; num_theta = num_particle_theta + num_fluid_theta + num_interface_theta;
  I1_q1 = 0; I2_q1 = I1_q1 + num_particle_q; 
  I1_q2 = I2_q1; I2_q2 = I1_q2 + num_fluid_phi; 
  I1_p1 = I2_q2; I2_p1 = I1_p1 + num_particle_p; 
  I1_p2 = I2_p1; I2_p2 = I1_p2 + num_fluid_p;
  I1_theta1 = I2_p2; I2_theta1 = I1_theta1 + num_particle_theta; # particle theta 
  I1_theta2 = I2_theta1; I2_theta2 = I1_theta2 + num_fluid_theta; # fluid theta 
  I1_theta3 = I2_theta2; I2_theta3 = I1_theta3 + num_interface_theta; # interface theta 

  I1_q = I1_q1; I2_q = I2_q2; 
  I1_p = I1_p1; I2_p = I2_p2; 
  I1_theta = I1_theta1; I2_theta = I2_theta3;

  I_in = {'i1':[I1_particle_q,I1_fluid_phi,I1_particle_p,I1_fluid_p,I1_particle_theta,I1_fluid_theta,I1_interface_theta],
          'i2':[I2_particle_q,I2_fluid_phi,I2_particle_p,I2_fluid_p,I2_particle_theta,I2_fluid_theta,I2_interface_theta]};
  I_local_in = {'i1':[I1_q1,I1_q2,I1_p1,I1_p2,I1_theta1,I1_theta2,I1_theta3],
                'i2':[I2_q1,I2_q2,I2_p1,I2_p2,I2_theta1,I2_theta2,I2_theta3]};
  I_local_out = {'i1':[I1_q1,I1_q2,I1_p1,I1_p2,I1_theta1,I1_theta2,I1_theta3],
                 'i2':[I2_q1,I2_q2,I2_p1,I2_p2,I2_theta1,I2_theta2,I2_theta3]};
  I_out = {'i1':[I1_particle_q,I1_fluid_phi,I1_particle_p,I1_fluid_p,I1_particle_theta,I1_fluid_theta,I1_interface_theta],
           'i2':[I2_particle_q,I2_fluid_phi,I2_particle_p,I2_fluid_p,I2_particle_theta,I2_fluid_theta,I2_interface_theta]};

  bar_K_j = np.zeros((num_q + num_p + num_theta, num_q + num_p + num_theta)); 
  dot_q = np.hstack((particle_p/m,(fluid_p/rho)));
  dot_q_fld_dx = np.hstack((particle_p/m,(fluid_p/rho)*deltaV)); # deltaV on dot_q for fluid energy density
 
  extras_D_j_interface = {'flag_save':True};
  D_j = compute_D_j_interface(Y,params,extras_D_j_interface); ii = c_v_I;
  if flag_save: D_j_list.append(D_j);
  partial_theta_I_u_j = partial_theta_j_u_j = c_v[ii['interface']];
  
  # interface friction terms 
  theta_I = interface_theta; 
  bar_K_j[I1_p:I2_p,I1_p:I2_p] = theta_I*D_j; 
  bar_K_j[I1_p:I2_p,I1_theta3:I2_theta3] = np.expand_dims(-theta_I*np.dot(D_j,dot_q_fld_dx)/partial_theta_I_u_j,1); 
  bar_K_j[I1_theta3:I2_theta3,I1_p:I2_p] = np.transpose(bar_K_j[I1_p:I2_p,I1_theta3:I2_theta3]);
  #bar_K_j[j,I1_theta3:I2_theta3,I1_theta3:I2_theta3] = theta_I*np.dot(dot_q,np.dot(D_j,dot_q))/(partial_theta_I_u_j*partial_theta_I_u_j);
  bar_K_j[I1_theta3:I2_theta3,I1_theta3:I2_theta3] = np.dot(dot_q_fld_dx,-1.0*bar_K_j[I1_p:I2_p,I1_theta3:I2_theta3])/partial_theta_I_u_j;

  # heat exchnage terms between the interface, fluid, and particle(s)   
  c_P = cc_P = c_v[ii['particle']]; c_F = cc_F = c_v[ii['fluid']]; 
  c_I = cc_I = c_v[ii['interface']]; 
  c_I_inv_dx = cc_I_inv_dx = cc_I/deltaV; # arises since specific heat of c_I per unit volume needed for fluid exchange
  c_F_inv_dx = cc_F_inv_dx = cc_F/deltaV; # arises since specific heat of c_F per unit volume needed for fluid exchange
  c_I_I = partial_tau_u_I_I = cc_I*cc_I; c_I_I_inv_dx = cc_I*cc_I_inv_dx;
  c_F_F = partial_tau_u_F_F = cc_F*cc_F; c_F_F_inv_dx = cc_F*cc_F_inv_dx;
  c_P_P = partial_tau_u_P_P = cc_P*cc_P; c_P_I = partial_tau_u_P_I = cc_P*cc_I; 
  c_F_I = partial_tau_u_F_I = cc_F*cc_I; c_F_I_inv_dx = cc_F*cc_I_inv_dx;
  theta_P = particle_theta; theta_F = fluid_theta; theta_I = interface_theta; 
  Gamma_op_vec = matrix_Gamma_op = extras_D_j_interface['matrix_Gamma_op'];
  aa = Gamma_op_vec.reshape(num_particles,num_dim,num_mesh_pts,num_dim);
  Gamma_op = Gamma_op_scalar = aa[:,0,:,0].reshape(num_particles,num_mesh_pts); # get scalar operator
  Lambda_op = Gamma_op.T;
  #kappa_F_I_dx = kappa_F_I*Lambda_op.flatten();  # spatial dependence of thermal conductivity
  kappa_F_I_xx_dx = kappa_F_I*Lambda_op.flatten();  # spatial dependence of thermal conductivity
  kappa_F_I_xx = kappa_F_I_xx_dx/deltaV;  # spatial dependence of thermal conductivity (so integrates)
  # these scalings follow, since we want conservation of internal energies in heat exchanges 
  # E[theta_F,theta_I] = sum_m c_F*theta_F(x_m) deltaV + c_I*theta_I.
  
  bar_K_j[I1_theta1:I2_theta1,I1_theta1:I2_theta1] = kappa_P_I*theta_I*theta_P/c_P_P; 
  bar_K_j[I1_theta1:I2_theta1,I1_theta3:I2_theta3] = -kappa_P_I*theta_P*theta_I/c_P_I;
  ii1 = range(I1_theta2,I2_theta2); ii2 = range(I1_theta2,I2_theta2);
  bar_K_j[ii1,ii2] = kappa_F_I_xx*theta_F*theta_I/(c_F_F*deltaV); # diagonal  
  bar_K_j[I1_theta2:I2_theta2,I1_theta3:I2_theta3] = np.expand_dims(-kappa_F_I_xx*theta_I*theta_F/c_F_I,1);
  bar_K_j[I1_theta3:I2_theta3,I1_theta1:I2_theta1] = -kappa_P_I*theta_I*theta_P/c_P_I;
  bar_K_j[I1_theta3:I2_theta3,I1_theta2:I2_theta2] = bar_K_j[I1_theta2:I2_theta2,I1_theta3:I2_theta3].transpose();
  bar_K_j[I1_theta3:I2_theta3,I1_theta3:I2_theta3] \
    += ((kappa_P_I*theta_P*theta_I)/c_I_I) + (np.sum(kappa_F_I_xx*theta_I*theta_F*deltaV)/c_I_I); # summing energy density, hence deltaV.
 
  # save the operator 
  bar_K_j_list.append(bar_K_j);
  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  # check the energy exchanges (kinetic to heat energy)

  if flag_save_energy_flux or flag_flux_check:
    rate_kinetic_particle = np.sum(bar_K_j[I1_p1:I2_p1,I1_theta3:I2_theta3].T*(1.0/theta_I)*(particle_p/m))*c_I;
    rate_kinetic_fluid = np.sum(bar_K_j[I1_p2:I2_p2,I1_theta3:I2_theta3].T*(1.0/theta_I)*(fluid_p/rho))*c_I*deltaV;
    rate_kinetic_total = rate_kinetic_particle + rate_kinetic_fluid;
    rate_heat_particle = np.sum(bar_K_j[I1_theta1:I2_theta1,I1_theta3:I2_theta3].T*(1.0/theta_I))*c_I*c_P;
    rate_heat_fluid = np.sum(bar_K_j[I1_theta2:I2_theta2,I1_theta3:I2_theta3].T*(1.0/theta_I))*c_I*c_F*deltaV; # sum energy density (deltaV)
    rate_heat_interface = np.sum(bar_K_j[I1_theta3:I2_theta3,I1_theta3:I2_theta3].T*(1.0/theta_I))*c_I*c_I;
    rate_heat_total = rate_heat_particle + rate_heat_fluid + rate_heat_interface;
    rate_total = rate_kinetic_total + rate_heat_total; 

  if flag_save_energy_flux:
    energy_flux = {};
    energy_flux['rate_kinetic_particle'] = rate_kinetic_particle;
    energy_flux['rate_kinetic_fluid'] = rate_kinetic_fluid;
    energy_flux['rate_kinetic_total'] = rate_kinetic_total;
    energy_flux['rate_heat_particle'] = rate_heat_particle;
    energy_flux['rate_heat_fluid'] = rate_heat_fluid;
    energy_flux['rate_heat_interface'] = rate_heat_interface;
    energy_flux['rate_heat_total'] = rate_heat_total;
    energy_flux['rate_total'] = rate_total;
    energy_flux_list.append(energy_flux); 

  # check the energy exchanges (kinetic to heat energy)
  if flag_flux_check:
    print("--");
    print("checking energy exchanges for interface:"); 
    print("rate_kinetic_particle = %.3e"%rate_kinetic_particle);
    print("rate_kinetic_fluid = %.3e"%rate_kinetic_fluid);
    print("rate_kinetic_total = %.3e"%rate_kinetic_total);
    print("rate_heat_particle = %.3e"%rate_heat_particle); 
    print("rate_heat_fluid = %.3e"%rate_heat_fluid); 
    print("rate_heat_interface = %.3e"%rate_heat_interface); 
    print("rate_heat_total = %.3e"%rate_heat_total); 
    print("rate_total = %.1e"%rate_total);
    print("--");
    print(""); 

  # package the results to return  
  bar_K_j_indices = {'I_in':I_in_list,'I_out':I_out_list,
                     'I_local_in':I_local_in_list,'I_local_out':I_local_out_list};

  if flag_save:
    extras.update({'D_j_list':D_j_list}); 

  if flag_save_energy_flux:
    extras.update({'energy_flux_list':energy_flux_list}); 
    
  return bar_K_j_list, bar_K_j_indices; 

def compute_div_K_j(Y,params=None,extras=None):
  m,rho,gamma,mu,Y_I,c_v,c_v_I = tuple(map(
    params.get,['m','rho','gamma','mu','Y_I','c_v','c_v_I']));
  num_mesh_x,num_mesh_y,deltaX = tuple(map(
    params.get,['num_mesh_x','num_mesh_y','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX; 
  num_mesh_pts = num_mesh_x*num_mesh_y;
  kappa_P_I,kappa_F_I = tuple(map(
    params.get,['kappa_P_I','kappa_F_I']));
  num_dim = params['num_dim'];
  num_dim_sq = num_dim*num_dim;
  flag_incompressible = params['flag_incompressible'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_p,particle_theta = tuple(map(
    dd.get,['particle_q','particle_p','particle_theta']));
  fluid_phi,fluid_p,fluid_theta = tuple(map(
    dd.get,['fluid_phi','fluid_p','fluid_theta']));
  interface_q,interface_p,interface_theta = tuple(map(
    dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_p,I2_particle_p = tuple(map(
    dd.get,['I1_particle_p','I2_particle_p']));
  I1_particle_theta,I2_particle_theta = tuple(map(
    dd.get,['I1_particle_theta','I2_particle_theta']));

  I1_fluid_phi,I2_fluid_phi = tuple(map(
    dd.get,['I1_fluid_phi','I2_fluid_phi']));
  I1_fluid_p,I2_fluid_p = tuple(map(
    dd.get,['I1_fluid_p','I2_fluid_p']));
  I1_fluid_theta,I2_fluid_theta = tuple(map(
    dd.get,['I1_fluid_theta','I2_fluid_theta']));

  I1_interface_q,I2_interface_q = tuple(map(
    dd.get,['I1_interface_q','I2_interface_q']));
  I1_interface_p,I2_interface_p = tuple(map(
    dd.get,['I1_interface_p','I2_interface_p']));
  I1_interface_theta,I2_interface_theta = tuple(map(
    dd.get,['I1_interface_theta','I2_interface_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_p = particle_p.shape[0];
  num_particle_theta = particle_theta.shape[0];
  num_fluid_phi = fluid_phi.shape[0]; num_fluid_p = fluid_p.shape[0];
  num_fluid_theta = fluid_theta.shape[0];
  num_interface_q = interface_q.shape[0]; 
  num_interface_p = interface_p.shape[0]; 
  num_interface_theta = interface_theta.shape[0];

  num_particles = num_particle_theta; 

  time_index = params['time_index'];
  #if time_index == 0:
  #  print("WARNING: div_K_j may be non-zero (not implemented yet)");

  #if bar_K_j is None;
  #  bar_K_j = compute_bar_K_j(Y,params);
 
  if extras is not None:
    D_j_particle,D_j_interface = tuple(map(
      extras.get,['D_j_particle','D_j_interface']));
    dot_F,partial_p_dot_F = tuple(map(extras.get,['dot_F','partial_p_dot_F']));
    matrix_tensor_div,matrix_tensor_grad,flag_save = tuple(map(
      extras.get,['matrix_tensor_div','matrix_tensor_grad','flag_save']));
    if flag_incompressible:
      matrix_wp, = tuple(map(extras.get,['matrix_wp']));
    K_heat, = tuple(map(extras.get,['K_heat']));
    K_visco_dot_F,K_visco_grad = tuple(map(extras.get,['K_visco_dot_F','K_visco_grad']));
    partial_theta_K_visco_dot_F, = tuple(map(
      extras.get,['partial_theta_K_visco_dot_F']));
    partial_theta_K_heat, = tuple(map(
      extras.get,['partial_theta_K_heat']));
    matrix_Lambda_op, = tuple(map(
      extras.get,['matrix_Lambda_op']));
  else:
    D_j_particle,D_j_interface,dot_F,partial_p_dot_F = (None,None,None,None); 
    matrix_tensor_div,matrix_tensor_grad,flag_save = (None,None,None);
    if flag_incompressible:
      matrix_wp = None;
    K_visco_dot_F,K_visco_grad,K_heat = (None,None,None);
    partial_theta_K_visco_dot_F = None;
    partial_theta_K_heat = None;
    matrix_Lambda_op = None;

  if D_j_particle is None:
    D_j_particle = compute_D_j_particle(Y,params,extras);

  if D_j_interface is None:
    D_j_interface = compute_D_j_interface(Y,params,extras);

  if matrix_tensor_div is None:
    matrix_tensor_div = compute_matrix_tensor_div(Y,params);

  if matrix_tensor_grad is None:
    matrix_tensor_grad = compute_matrix_tensor_grad(Y,params);

  if flag_incompressible and (matrix_wp is None):
    matrix_wp = compute_matrix_wp(Y,params);

  if K_visco_dot_F is None:
    K_visco_dot_F = compute_K_visco_dot_F(Y,params,extras);

  if K_visco_grad is None:
    K_visco_grad = compute_K_visco_grad(Y,params,extras);

  if K_heat is None:
    K_heat = compute_K_heat2(Y,params,extras);

  if dot_F is None:
    dot_F = compute_dot_F(Y,params);

  if partial_p_dot_F is None:
    partial_p_dot_F = compute_partial_p_dot_F(Y,params); 
 
  if partial_theta_K_visco_dot_F is None:
    partial_theta_K_visco_dot_F = compute_partial_theta_K_visco_dot_F(Y,params,extras);

  if partial_theta_K_heat is None:
    partial_theta_K_heat = compute_partial_theta_K_heat2(Y,params,extras);

  if matrix_Lambda_op is None:
    matrix_Lambda_op = compute_matrix_Lambda_op(Y,params);

  div_K_j_list = [];
  ii = c_v_I; 

  j = ii['particle']; c_P = c_v[j];
  div_K_j = np.zeros(num_particle_p + num_particle_theta);
  D_j = D_j_particle; 
  i1 = 0; i2 = i1 + num_particle_p;
  dot_q = particle_p/m;   
  np_dot__D_j_dot_q = np.dot(D_j,dot_q);
  div_K_j[i1:i2] = -np_dot__D_j_dot_q/c_P;
  i1 = i2; i2 = i1 + num_particle_theta;  
  div_K_j[i1:i2] = -particle_theta*np.trace(D_j)/(m*c_P);
  i1 = i2; i2 = i1 + num_particle_theta;  
  div_K_j[i1:i2] += np.dot(dot_q,np_dot__D_j_dot_q)/(c_P*c_P);
  div_K_j_list.append(div_K_j);

  j = ii['fluid']; c_F = c_v[j];
  div_K_j = np.zeros(num_fluid_p + num_fluid_theta);
  i1 = 0; i2 = i1 + num_fluid_p;
  theta_F = fluid_theta;
  # The operator is given by action of  
  # $\frac{\mbox{\small div}\left(K_{visco} \square \dot{\mb{F}} \right)}{\partial_\tau u}$
  # need action of operator for multiplication by 1/theta_F(x_m) at each x-location 
  KK = K_visco_dot_F.reshape(num_mesh_pts,num_dim_sq);
  one_over_theta_F = np.expand_dims((1.0/theta_F),1);
  partial_theta_K_visco_square_dot_F = (KK*one_over_theta_F).flatten();
  aa = np.dot(matrix_tensor_div,partial_theta_K_visco_square_dot_F);
  if flag_incompressible: # multiply final result by $\wp$
    aa = np.matmul(matrix_wp,aa); # additional terms: \wp (--)
    aa = conv_real(aa); 
  div_partial_theta_K_visco_square_dot_F = aa; 
  div_K_j[i1:i2] = div_partial_theta_K_visco_square_dot_F/c_F; 

  i1 = i2; i2 = i1 + num_fluid_theta; 
  # the tensor-product for each temperature at each x-location
  # added a second contraction in ee to reduce the vector dependence (but double-check)
  # (seems would need a factor fo deltaV when summing to get integrals)
  # The operator is given by action of  
  aa = K_visco_grad;
  aaa = aa.reshape(num_mesh_pts,num_dim_sq,num_mesh_pts*num_dim); 
  # grad.shape = num_mesh_pts,num_dim_sq,num_mesh_pts,num_dim
  gg = matrix_tensor_grad;
  ggg = gg.reshape(num_mesh_pts,num_dim_sq,num_mesh_pts*num_dim);
  bbb = np.sum(aaa*ggg,(1,2)); 
  partial_p_dot_F_colin_K_visco_grad = (1.0/rho)*bbb;

  div_K_j[i1:i2] += -partial_p_dot_F_colin_K_visco_grad/c_F;

  #bar_K_j[I1_p:I2_p,I1_theta:I2_theta] = div_K_visco_square_dot_F/partial_theta_j_u_j;

  # compute other contributions
  aa = dot_F;
  aaa = aa.reshape(num_mesh_pts,num_dim_sq);
  kk = partial_theta_K_visco_dot_F;
  kkk = kk.reshape(num_mesh_pts,num_dim_sq);
  div_K_j[i1:i2] += np.sum(aaa*kkk,1)/(c_F*c_F);

  #for dd in range(0,num_dim_sq):
  #  div_K_j[i1:i2] += (np.sum(dot_F[dd::num_dim_sq]*partial_theta_K_visco_dot_F[dd::num_dim_sq])/(c_F*c_F)); 

  div_K_j[i1:i2] += partial_theta_K_heat/(c_F*c_F);

  div_K_j_list.append(div_K_j);  

  j = ii['interface']; 
  # interface terms between the interface, fluid, and particle(s)   
  c_P = c_v[ii['particle']]; c_F = c_v[ii['fluid']]; c_I = c_v[ii['interface']];
  c_I_I = partial_tau_u_I_I = c_I*c_I; c_F_F = partial_tau_u_F_F = c_F*c_F;
  c_P_P = partial_tau_u_P_P = c_P*c_P; c_P_I = partial_tau_u_P_I = c_P*c_I; 
  c_F_I = partial_tau_u_F_I = c_F*c_I;
  theta_P = particle_theta; theta_F = fluid_theta; theta_I = interface_theta; 
  Lambda_scalar_op = matrix_Lambda_op[0::num_dim,0]; # get the scalar component of Lambda operator 
  kappa_F_I_xx_dx = kappa_F_I*Lambda_scalar_op.flatten(); # spatial dependence of thermal conductivity
  kappa_F_I_xx = kappa_F_I_xx_dx/deltaV;  # spatial dependence of thermal conductivity (so integrates)

  i1_particle_p = 0; i2_particle_p = i1_particle_p + num_particle_p;
  i1_fluid_p = i2_particle_p; i2_fluid_p = i1_fluid_p + num_fluid_p; 
  i1_particle_theta = i2_fluid_p; 
  i2_particle_theta = i1_particle_theta + num_particle_theta; 
  i1_fluid_theta = i2_particle_theta; 
  i2_fluid_theta = i1_fluid_theta + num_fluid_theta;
  i1_interface_theta = i2_fluid_theta; 
  i2_interface_theta = i1_interface_theta + num_interface_theta; 

  # setup divergence vector
  nn = i2_interface_theta;
  div_K_j = np.zeros(nn);

  # == 
  # dissipation friction terms (handle first, then other heat exchanges)
  D_j = D_j_interface; 
  dot_q = np.zeros(num_particle_p + num_fluid_p);
  dot_q_fld_dx = np.zeros(num_particle_p + num_fluid_p);
  ii1 = 0; ii2 = ii1 + num_particle_p; 
  dot_q[ii1:ii2] = particle_p/m; 
  dot_q_fld_dx[ii1:ii2] = particle_p/m; 
  ii1 = ii2; ii2 = ii1 + num_fluid_p; 
  dot_q[ii1:ii2] = (fluid_p/rho); 
  dot_q_fld_dx[ii1:ii2] = (fluid_p/rho)*deltaV; # @@@ deltaV in dot_q def 

  # particle + fluid contributions 
  i1 = i1_particle_p; i2 = i1 + num_particle_p + num_fluid_p;
  np_dot__D_j_dot_q = np.dot(D_j,dot_q_fld_dx);
  div_K_j[i1:i2] += -np_dot__D_j_dot_q/c_I;

  # trace term: first part
  i1 = i1_interface_theta; 
  i2 = i1 + num_interface_theta;
  ii1 = 0; ii2 = ii1 + num_particle_p; 
  di = (range(ii1,ii2),range(ii1,ii2)); # select diagonal indices
  div_K_j[i1:i2] += -interface_theta*np.sum(D_j[di])/(m*c_I);  

  # trace term: second part 
  i1 = i1_interface_theta; 
  i2 = i1 + num_interface_theta;
  ii1 = ii2; ii2 = ii1 + num_fluid_p; 
  di = (range(ii1,ii2),range(ii1,ii2)); # select diagonal indices
  div_K_j[i1:i2] += -interface_theta*np.sum(D_j[di])*(deltaV/(rho*c_I)); # deltaV from density and dot_q_fld_dx

  # q^T D q/C_I_I term 
  i1 = i1_interface_theta; 
  i2 = i1 + num_interface_theta;
  div_K_j[i1:i2] += np.dot(dot_q_fld_dx,np_dot__D_j_dot_q)/(c_I_I);
 
  # ==
  # heat exchange:
  # particle_theta
  i1 = i1_particle_theta; i2 = i1 + num_particle_theta;
  div_K_j[i1:i2] += kappa_P_I*theta_I/c_P_P;
  div_K_j[i1:i2] += -kappa_P_I*theta_P/c_P_I;

  # fluid_theta
  i1 = i1_fluid_theta; i2 = i1 + num_fluid_theta;
  div_K_j[i1:i2] += kappa_F_I_xx_dx*theta_I/(c_F_F*deltaV*deltaV);
  div_K_j[i1:i2] += -kappa_F_I_xx_dx*theta_F/(c_F_I*deltaV);

  # interface_theta
  i1 = i1_interface_theta; 
  i2 = i1 + num_interface_theta;
  div_K_j[i1:i2] += -kappa_P_I*theta_I/c_P_I;
  div_K_j[i1:i2] += kappa_P_I*theta_P/c_I_I;
  div_K_j[i1:i2] += -np.sum(kappa_F_I_xx_dx)*theta_I/(c_F_I*deltaV);
  div_K_j[i1:i2] += np.sum(kappa_F_I_xx_dx*theta_F)/c_I_I; 

  div_K_j_list.append(div_K_j);

  return div_K_j_list;  

def conv_vec_K_indexed_to_B_indexed(vec_a,Y,K_j_indices,B_j_indices):
  raise Exception("not implemented.");



def compute_div_K_j_monte_carlo(Y,params=None,**kargs):
  """ Monte-Carlo method for estimating the divergence of K_j """

  if kargs is not None: 
    num_samples,delta,flag_verbose = tuple(map(kargs.get,['num_samples','delta','flag_verbose']));
    flag_save_file,save_skip = tuple(map(kargs.get,['flag_save_file','save_skip']));
  else:
    num_samples,delta,flag_verbose = (None,None,None);
    flag_save_file,save_skip = (None,None);

  if num_samples is None:
    num_samples = int(1e1);

  if delta is None:
    delta = 1e-2;

  if flag_verbose is None:
    flag_verbose = 0; 

  if flag_save_file is None:
    flag_save_file = False; 

  if save_skip is None:
    save_skip = num_samples//10; 

  # monte-carlo sample 
  # E[(K_j(Y + xi) - K_j(Y))xi] ~= div(K_j)
  # uses (Kp - K) ~ xii_ell*\partial_ell K_{ij} 
  # E[xii_ell*\partial_ell K_{ij} xii_j]/delta = delta_{ell,j} \partial_ell K_{ij} 
  num_heat_bodies = params['num_heat_bodies'];
  sqrt_delta = np.sqrt(delta);
  K_j_list, K_j_indices = compute_bar_K_j(Y,params); 
  B_j_list,B_j_indices = compute_B_j_factors(Y,params);
  div_K_j_list = [];
  div_K_j_global_list = [];
  skip_display = num_samples//20; 
  for k in range(0,num_samples):
    xi = sqrt_delta*np.random.randn(Y.shape[0]); Yp = Y + xi;
    Kp_j_list, Kp_j_indices = compute_bar_K_j(Yp,params); 
    if flag_verbose >= 2 and (k % skip_display == 0):
      print("k = %d"%k);
    for j in range(0,num_heat_bodies):
      I_out = K_j_indices['I_out'][j];
      I_local_out = K_j_indices['I_local_out'][j];
      K_j = K_j_list[j]; Kp_j = Kp_j_list[j];
      xii = np.zeros(K_j.shape[0]);
      if k == 0:
        div_K_j_list.append(np.zeros(K_j.shape[0]));
      div_K_j = div_K_j_list[j];
      for ll in range(0,len(I_out['i1'])):
        i1 = I_local_out['i1'][ll]; i2 = I_local_out['i2'][ll];
        ii1 = I_out['i1'][ll]; ii2 = I_out['i2'][ll];
        xii[i1:i2] = xi[ii1:ii2];
      # using (Kp - K) ~ xii_ell*\partial_ell K_{ij}
      # divide by delta below when averaging 
      div_K_j[:] += np.dot(Kp_j - K_j,xii);  

    if (k != 0) and (k % save_skip) == 0: # save intermediate results
      base_dir = params['base_dir'];
      filename = base_dir + '/debug/compute_div_K_j_monte_carlo_%.8d.pickle'%k;
      ss = {'params_mc':kargs,'div_K_j_list__internal_sum':div_K_j_list,'sample_k':k}; 
      print("filename = " + filename);
      fid = open(filename,"wb"); pickle.dump(ss,fid); fid.close();

  # average and divide by delta
  for j in range(0,num_heat_bodies):
    div_K_j_list[j] = div_K_j_list[j]/(delta*num_samples);

  # need to convert to B_j_indices for compatability 
  div_K_j_list2 = [];
  for j in range(0,num_heat_bodies): 
    I_out_k = K_j_indices['I_out'][j];
    I_local_out_k = K_j_indices['I_local_out'][j];
    I_out_b = B_j_indices['I_out'][j];
    I_local_out_b = B_j_indices['I_local_out'][j]; 
    div_K_j = div_K_j_list[j];
    div_K_j_global = np.zeros(Y.shape[0]);
    for ll in range(0,len(I_out_k['i1'])):
      i1 = I_out_k['i1'][ll]; i2 = I_out_k['i2'][ll];
      ii1 = I_local_out_k['i1'][ll]; ii2 = I_local_out_k['i2'][ll];
      div_K_j_global[i1:i2] = div_K_j[ii1:ii2];
    B_j = B_j_list[j];
    div_K_j_2 = np.zeros(B_j.shape[0]);
    for ll in range(0,len(I_out_b['i1'])):
      i1 = I_out_b['i1'][ll]; i2 = I_out_b['i2'][ll];
      ii1 = I_local_out_b['i1'][ll]; ii2 = I_local_out_b['i2'][ll];
      div_K_j_2[ii1:ii2] = div_K_j_global[i1:i2];

    div_K_j_list2.append(div_K_j_2);
   
  return div_K_j_list2;


def compute_B_j_factors(Y,params=None,extras=None):
  """
  Uses factorization of $bar_K_j = N_E K_0 N_E^*$ and 
  $bar_K_j = M_E K_0 M_E^*$ to compute the needed 
  factors for the noise generation $RR^T = K$ 
  """
  m,rho,gamma,mu,Y_I,c_v,c_v_I = tuple(map(
    params.get,['m','rho','gamma','mu','Y_I','c_v','c_v_I']));
  num_mesh_x,num_mesh_y,deltaX = tuple(map(
    params.get,['num_mesh_x','num_mesh_y','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX; 
  num_mesh_pts = num_mesh_x*num_mesh_y; 
  kappa_P_I,kappa_F_I = tuple(map(
    params.get,['kappa_P_I','kappa_F_I']));
  num_dim = params['num_dim']; num_dim_sq = num_dim*num_dim; 
  k_B = params['k_B'];
  sqrt_two_k_B = np.sqrt(2.0*k_B);
  flag_incompressible = params['flag_incompressible'];

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_p,particle_theta = tuple(map(
    dd.get,['particle_q','particle_p','particle_theta']));
  fluid_phi,fluid_p,fluid_theta = tuple(map(
    dd.get,['fluid_phi','fluid_p','fluid_theta']));
  interface_q,interface_p,interface_theta = tuple(map(
    dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_p,I2_particle_p = tuple(map(
    dd.get,['I1_particle_p','I2_particle_p']));
  I1_particle_theta,I2_particle_theta = tuple(map(
   dd.get,['I1_particle_theta','I2_particle_theta']));

  I1_fluid_phi,I2_fluid_phi = tuple(map(
    dd.get,['I1_fluid_phi','I2_fluid_phi']));
  I1_fluid_p,I2_fluid_p = tuple(map(
    dd.get,['I1_fluid_p','I2_fluid_p']));
  I1_fluid_theta,I2_fluid_theta = tuple(map(
    dd.get,['I1_fluid_theta','I2_fluid_theta']));

  I1_interface_q,I2_interface_q = tuple(map(
    dd.get,['I1_interface_q','I2_interface_q']));
  I1_interface_p,I2_interface_p = tuple(map(
    dd.get,['I1_interface_p','I2_interface_p']));
  I1_interface_theta,I2_interface_theta = tuple(map(
    dd.get,['I1_interface_theta','I2_interface_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_p = particle_p.shape[0];
  num_particle_theta = particle_theta.shape[0]; 
  num_fluid_phi = fluid_phi.shape[0]; num_fluid_p = fluid_p.shape[0]; 
  num_fluid_theta = fluid_theta.shape[0]; num_interface_q = interface_q.shape[0]; 
  num_interface_p = interface_p.shape[0]; 
  num_interface_theta = interface_theta.shape[0];

  if extras is not None:
    bar_K_j,bar_K_j_indices,matrix_tensor_div,flag_save = tuple(map(
      extras.get,['bar_K_j','bar_K_j_indices','matrix_tensor_div','flag_save']));
    extras_matrix_tensor_div,extras_K_visco_dot_F = tuple(map(
      extras.get,['extras_matrix_tensor_div','extras_K_visco_dot_F']));
    extras_matrix_wp, = tuple(map(
      extras.get,['extras_matrix_wp']));
    matrix_Lambda_op, = tuple(map(
      extras.get,['matrix_Lambda_op']));
  else:
    bar_K_j,bar_K_j_indices,matrix_tensor_div,flag_save = (None,None,None,None);
    extras_matrix_tensor_div,extras_K_visco_dot_F = (None,None);
    extras_matrix_wp = None;
    matrix_Lambda_op = None;

  if flag_save is None:
    flag_save = False; 
 
  if bar_K_j is None:
    bar_K_j,bar_K_j_indices = compute_bar_K_j(Y,params,extras);
    if extras is not None:
      extras_matrix_tensor_div = extras['extras_matrix_tensor_div'];
      extras_matrix_wp = extras['extras_matrix_wp'];
      extras_K_visco_dot_F = extras['extras_K_visco_dot_F'];

  extras_R_visco = {};
  if extras_matrix_tensor_div is not None:
    extras_R_visco.update({'matrix_tensor_div':extras_matrix_tensor_div['D'],
                           'extras_K_visco_dot_F':extras_K_visco_dot_F});
    matrix_tensor_div = extras_matrix_tensor_div['D'];
  R_visco = compute_R_visco(Y,params,extras_R_visco);

  extras_R_heat = {};
  if extras_matrix_tensor_div is not None:
    extras_R_heat.update({'matrix_tensor_div':extras_matrix_tensor_div['D']});
  if extras_K_visco_dot_F is not None:
    extras_R_heat.update({'dot_F':extras_K_visco_dot_F['dot_F']});
    dot_F = extras_K_visco_dot_F['dot_F'];
  else:
    dot_F = None; 

  if extras_matrix_wp is not None:
    matrix_wp = extras_matrix_wp['matrix_wp'];
  else:
    matrix_wp = None; 

  R_heat = compute_R_heat2(Y,params,extras_R_heat);

  if dot_F is None:
    dot_F = compute_dot_F(Y,params);

  if matrix_tensor_div is None:
    matrix_tensor_div = compute_matrix_tensor_div(Y,params);

  if matrix_wp is None:
    matrix_wp = compute_matrix_wp(Y,params);

  if matrix_Lambda_op is None:
    matrix_Lambda_op = compute_matrix_Lambda_op(Y,params);

  Lambda_scalar_op = matrix_Lambda_op[0::num_dim,0]; # get the scalar component of Lambda operator

  # debugging
  #print("WARNING: set kappa_F_I = 0, kappa_P_I = 0");
  #kappa_F_I = 0; kappa_P_I = 0;   

  kappa_F_I_xx_dx = kappa_F_I*Lambda_scalar_op;  # spatial dependence of thermal conductivity
  #kappa_F_I_xx = kappa_F_I_xx_dx/deltaV;  # spatial dependence of thermal conductivity (so integrates)

  B_j_list = []; 
  I_in_list = []; I_out_list = []; 
  I_local_in_list = []; I_local_out_list = []; 
  ii = c_v_I;

  # compute factors 
  # WARNING: particle-case not implemented currently.
  # == 
  j = ii['particle']; 
  n1 = num_particle_p + num_particle_theta;
  n2 = num_particle_p + num_particle_theta;
  #B_j = sqrt_two_k_B*np.zeros((n1,n2));
  B_j = np.zeros((n1,n2));

  I_in = None; # not relevant for noise 
  I1_p = 0; I2_p = I1_p + num_particle_p;
  I1_theta = I2_p; I2_theta = I1_theta + num_particle_theta;
  I_local_in = None; # not relevant for noise 
  I_local_out = {'i1':[I1_p,I1_theta],'i2':[I2_p,I2_theta]};  
  I_out = {'i1':[I1_particle_p,I1_particle_theta],'i2':[I2_particle_p,I2_particle_theta]};

  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  B_j = sqrt_two_k_B*B_j; # scale so BB^T = 2*k_B*K 
  B_j_list.append(B_j);

  # ==
  j = ii['fluid']; c_F = c_v[j]; 
  n1 = num_fluid_p + num_fluid_theta;
  n2 = num_mesh_x*num_mesh_y*num_dim_sq + num_fluid_theta*num_dim;
  B_j = np.zeros((n1,n2));
  theta_F = fluid_theta;

  # use the computed R_visco and R_heat to construct B_j
  i1 = 0; i2 = i1 + num_mesh_x*num_mesh_y*num_dim;
  j1 = 0; j2 = j1 + num_mesh_x*num_mesh_y*num_dim_sq;
  aa = -np.dot(matrix_tensor_div,R_visco);
  if flag_incompressible:
    aa = np.matmul(matrix_wp,aa); 
  B_j[i1:i2,j1:j2] = aa; 
  i2_theta_start = i2;

  i1 = i2_theta_start; i2 = i1 + num_fluid_theta;
  j1 = 0; j2 = j1 + num_mesh_pts*num_dim_sq;
  FF = dot_F.reshape(num_mesh_pts,num_dim_sq,1,1); 
  A = R_visco/c_F; AA = A.reshape(num_mesh_pts,num_dim_sq,num_mesh_pts,num_dim_sq);
  CC = -np.sum(FF*AA,1); # contraction ->  -F:R_visco/c_F.
  C = CC.reshape(num_mesh_pts,num_mesh_pts*num_dim_sq);
  B_j[i1:i2,j1:j2] += C;

  i1 = i2_theta_start; i2 = i1 + num_fluid_theta;
  j1 = j2; j2 = j1 + num_fluid_theta*num_dim;
  B_j[i1:i2,j1:j2] += R_heat/c_F;

  I_in = None; # note relevant for noise
  I1_p = 0; I2_p = I1_p + num_fluid_p;
  I1_theta = I2_p; I2_theta = I1_theta + num_fluid_theta;
  I_local_in = None; # not relevant for noise 
  I_local_out = {'i1':[I1_p,I1_theta],'i2':[I2_p,I2_theta]};  
  I_out = {'i1':[I1_fluid_p,I1_fluid_theta],'i2':[I2_fluid_p,I2_fluid_theta]};

  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  B_j = sqrt_two_k_B*B_j; # scale so BB^T = 2*k_B*K 
  B_j_list.append(B_j);
 
  # ==
  j = ii['interface']; c_I = c_v[j];
  # we break down the generation of fluctuations into parts 
  # so that g = R_1 x_1 + R_2 x_2.  This can be combined into
  # one large B matrix of the form B=[R_1|R_2] for xi = [xi_1|x_2]^T.
  n1 = num_particle_p + num_fluid_p + num_particle_theta + num_fluid_theta + num_interface_theta;
  n2 = num_particle_p + num_particle_theta + num_fluid_theta;  
 
  i1_particle_p = 0; i2_particle_p = i1_particle_p + num_particle_p;
  i1_fluid_p = i2_particle_p; i2_fluid_p = i1_fluid_p + num_fluid_p; 
  i1_particle_theta = i2_fluid_p; 
  i2_particle_theta = i1_particle_theta + num_particle_theta; 
  i1_fluid_theta = i2_particle_theta; 
  i2_fluid_theta = i1_fluid_theta + num_fluid_theta;
  i1_interface_theta = i2_fluid_theta; 
  i2_interface_theta = i1_interface_theta + num_interface_theta; 

  i1_B_theta_P = num_particle_p + num_fluid_p;
  i1_B_theta_F = num_particle_p + num_fluid_p + num_particle_theta;
  i1_B_theta_I = num_particle_p + num_fluid_p + num_particle_theta + num_fluid_theta;

  B_j = np.zeros((n1,n2));
  
  # --  
  # dissipation terms R_1 
  n1 = num_particle_p + num_fluid_p + num_interface_theta;
  n2 = num_particle_p;

  i1_R1_theta_I = num_particle_p + num_fluid_p;

  R_1 = np.zeros((n1,n2));
 
  dot_q = np.zeros((num_particle_p + num_fluid_p)); 
  dot_q_fld_dx = np.zeros((num_particle_p + num_fluid_p)); 
  i1 = 0; i2 = i1 + num_particle_p;
  dot_q[i1:i2] = particle_p/m;
  dot_q_fld_dx[i1:i2] = particle_p/m;
  i1 = i2; i2 = i1 + num_fluid_p;
  dot_q[i1:i2] = (fluid_p/rho);
  dot_q_fld_dx[i1:i2] = (fluid_p/rho)*deltaV;

  R_D = np.zeros((num_particle_p + num_fluid_p,num_particle_p));
  i1 = 0; i2 = i1 + num_particle_p;
  j1 = 0; j2 = j1 + num_particle_p;
  R_D[i1:i2,j1:j2] = np.sqrt(gamma)*np.eye(num_particle_p); 
  i1 = i2; i2 = i1 + num_fluid_p;
  j1 = 0; j2 = j1 + num_particle_p;
  R_D[i1:i2,j1:j2] = -np.sqrt(gamma)*matrix_Lambda_op/deltaV; # @@@ deltaV for scaling 

  # particle_p-fluid_p rows
  i1 = 0; i2 = i1 + num_particle_p + num_fluid_p;
  j1 = 0; j2 = j1 + num_particle_p;
  R_1[i1:i2,j1:j2] = np.sqrt(interface_theta)*R_D;
 
  # theta_I row 
  i1 = i1_R1_theta_I; i2 = i1 + num_interface_theta;
  j1 = 0; j2 = j1 + num_particle_p;
  R_1[i1:i2,j1:j2] = -np.sqrt(interface_theta)*np.dot(dot_q_fld_dx,R_D)/c_I;

  # copy R_1 into B_j components (particle_p-fluid_p rows)
  i1 = 0; i2 = i1 + num_particle_p + num_fluid_p;
  j1 = 0; j2 = j1 + num_particle_p;
  ii1 = 0; ii2 = ii1 + num_particle_p + num_fluid_p;
  jj1 = 0; jj2 = jj1 + num_particle_p;
  B_j[i1:i2,j1:j2] = R_1[ii1:ii2,jj1:jj2];  

  # copy R_1 into B_j components (theta_I row)
  i1 = i1_B_theta_I; i2 = i1 + num_interface_theta;
  j1 = 0; j2 = j1 + num_particle_p;
  ii1 = i1_R1_theta_I; ii2 = ii1 + num_interface_theta;
  jj1 = 0; jj2 = jj1 + num_particle_p;
  B_j[i1:i2,j1:j2] = R_1[ii1:ii2,jj1:jj2];  
  j2_B_R1 = j2;

  # --
  # heat exchange terms for P_I
  theta_P = particle_theta; theta_F = fluid_theta; theta_I = interface_theta;
  ii = c_v_I;
  c_P = c_v[ii['particle']]; c_F = c_v[ii['fluid']]; c_I = c_v[ii['interface']];

  i1_R21_theta_P = 0;
  i1_R21_theta_I = i1_R21_theta_P + num_particle_theta;

  nn_theta = num_particle_theta + num_interface_theta
  R_21 = np.zeros((nn_theta,num_particle_theta)); # is just a vector 

  R_21[0,0] = np.sqrt(kappa_P_I*theta_P[0]*theta_I[0])*(1.0/c_P); # WARNING: assume 1 particle  
  R_21[1,0] = np.sqrt(kappa_P_I*theta_P[0]*theta_I[0])*(-1.0/c_I); 
  
  # copy R_21 into B_j 
  i1 = i1_B_theta_P; i2 = i1 + num_particle_theta;
  j1 = j2_B_R1; j2 = j1 + num_particle_theta;
  ii1 = i1_R21_theta_P; ii2 = ii1 + num_particle_theta; 
  jj1 = 0; jj2 = jj1 + num_particle_theta;
  B_j[i1:i2,j1:j2] = R_21[ii1:ii2,jj1:jj2];  

  i1 = i1_B_theta_I; i2 = i1 + num_interface_theta;
  j1 = j2_B_R1; j2 = j1 + num_particle_theta;
  ii1 = i1_R21_theta_I; ii2 = ii1 + num_interface_theta; 
  jj1 = 0; jj2 = jj1 + num_particle_theta;
  B_j[i1:i2,j1:j2] = R_21[ii1:ii2,jj1:jj2];  
  j2_R21 = j2; 
  
  # heat exchange terms for F_I
  num_theta = 2;
  i1_R22_theta_F = 0; i1_R22_theta_I = 1;
  R_22_x = np.zeros((num_theta,num_mesh_pts)); # is just a two vector for each x location
  R_22_x[i1_R22_theta_F,:] = np.sqrt(kappa_F_I_xx_dx*theta_F*theta_I)*(1.0/(c_F*deltaV)); # scaling by deltaV 
  R_22_x[i1_R22_theta_I,:] = -np.sqrt(kappa_F_I_xx_dx*theta_F*theta_I)*(1.0/c_I); # set last row

  #R_22_x = R_22_x/deltaV; # @@@ deltaV scaling suggested by numerical studies 

  # copy into B_j 
  i1 = i1_B_theta_F; i2 = i1 + num_fluid_theta;
  ii_range = range(i1,i2); 
  j1 = j2_R21; j2 = j1 + num_fluid_theta;
  jj_range = range(j1,j2);
  B_j[ii_range,jj_range] += R_22_x[i1_R22_theta_F,:];

  #i1 = i1_B_theta_I; i2 = i1 + num_interface_theta;
  ii_range = i1_B_theta_I*np.ones(num_fluid_theta,dtype=int);
  j1 = j2_R21; j2 = j1 + num_fluid_theta;
  jj_range = range(j1,j2);
  B_j[ii_range,jj_range] += R_22_x[i1_R22_theta_I,:];  
 
  I_in = None; # not relevant since random noise generation 
  I_local_in = None; # note relevant since random noise generation 
  I1_p1 = 0; I2_p1 = I1_p1 + num_particle_p; 
  I1_p2 = I2_p1; I2_p2 = I1_p2 + num_fluid_p; 
  I1_theta1 = I2_p2; I2_theta1 = I1_theta1 + num_particle_theta;
  I1_theta2 = I2_theta1; I2_theta2 = I1_theta2 + num_fluid_theta;
  I1_theta3 = I2_theta2; I2_theta3 = I1_theta3 + num_interface_theta;
  I_local_out = {'i1':[I1_p1,I1_p2,I1_theta1,I1_theta2,I1_theta3],
                 'i2':[I2_p1,I2_p2,I2_theta1,I2_theta2,I2_theta3]};  
  I_out = {'i1':[I1_particle_p,I1_fluid_p,I1_particle_theta,I1_fluid_theta,I1_interface_theta],
           'i2':[I2_particle_p,I2_fluid_p,I2_particle_theta,I2_fluid_theta,I2_interface_theta]};

  I_in_list.append(I_in); I_out_list.append(I_out); 
  I_local_in_list.append(I_local_in); I_local_out_list.append(I_local_out);

  B_j = sqrt_two_k_B*B_j; # scale so BB^T = 2*k_B*K 
  B_j_list.append(B_j);

  # == 
  B_j_indices = {'I_in':I_in_list,'I_out':I_out_list,
                 'I_local_in':I_local_in_list,'I_local_out':I_local_out_list};

  if flag_save:
    extras.update({'matrix_tensor_div':matrix_tensor_div,'matrix_wp':matrix_wp});

  return B_j_list,B_j_indices; 
 

def compute_B_j_svd(Y,params=None,extras=None):
  """
  Uses SVD of bar_K_j to compute the factors for noise generation
  $RR^T = K$ 
  """
  num_dim = params['num_dim'];
  k_B = params['k_B'];
  sqrt_two_k_B = np.sqrt(2.0*k_B);

  if extras is not None:
    bar_K_j,bar_K_j_indices = tuple(map(extras.get,['bar_K_j','bar_K_j_indices']));
  else:
    bar_K_j,bar_K_j_indices = (None,None);

  if bar_K_j is None:
    bar_K_j,bar_K_j_indices = compute_bar_K_j(Y_n,params,extras);

  num_heat_bodies = params['num_heat_bodies']; # hard coded for particle, fluid, interface 
  B_j_list = []; B_j_indices_list = [];
  for j in range(0,num_heat_bodies):
    U,S,Vh = np.linalg.svd(bar_K_j[j],hermitian=True);
    sqrt_S = np.diag(np.sqrt(S));
    B_j_list.append(sqrt_two_k_B*np.matmul(U,sqrt_S));
    B_j_indices_list.append(K_j_indices[j]);

  return B_j_list,B_j_indices_list;


def compute_g_thm_j_dt(Y_n,params,extras=None):
  # Compute fluctuation constributions.
  #
  # For a summary, see pg. 26 of 
  # H. Ottinger book, Beyond Eqilibrium.
  #
  # We combine this with our factorizations
  # and other results. 
  #
  # g_thm dt =  k_B*div_Y(K)dt + B(Y)*dW_t,
  # where, 
  # B(Y)B(Y)^T = 2*k_B K(Y).
  #

  num_dim,k_B,Y_I,deltaT,flag_save_B_j_tensors,flag_compute_div_K = tuple(map(params.get,
    ['num_dim','k_B','Y_I','deltaT','flag_save_B_j_tensors','flag_compute_div_K']));

  if num_dim is None or k_B is None or Y_I is None \
     or deltaT is None or flag_compute_div_K is None:

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

  B_j_list,B_j_indices = compute_B_j_factors(Y_n,params,extras); ii = Y_I;

  if flag_compute_div_K:
    div_K_j_list = compute_div_K_j(Y_n,params,extras);
    if flag_save_div_K_j:
      extras.update({'div_K_j_list':div_K_j_list});

  if flag_save_B_j_tensors:
    extras.update({'B_j':B_j_list,'B_j_indices':B_j_indices});
    if flag_compute_div_K:
      extras.update({'div_K_j_list':div_K_j_list});

  g_thm_j_dt_list = [];

  if flag_use_saved_dW:
    dW_list = extras['dW_list'];
  elif flag_save_dW:
    dW_list = [];

  sqrt_deltaT = np.sqrt(deltaT);
  num_heat_bodies = params['num_heat_bodies']; # particle, conc, interface 

  # generate the stochastic forces 
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

# debug
# print divergence tensor 
def pp_div(matrix_D,ii):
  for k in range(0,25):
    print("k:%.2d"%k + " " + str(matrix_D[ii,k*num_dim*num_dim:(k+1)*num_dim*num_dim]));
    if (k + 1) % 5 == 0:
      print("")

# print divergence tensor 
def pp_grad(matrix_G,ii):
  for k in range(0,25):
    print("k:%.2d"%k + " " + str(matrix_G[ii,k*num_dim:(k+1)*num_dim]));
    if (k + 1) % 5 == 0:
      print("")


def test_B_j(Y,params,extras=None):
  B_j_list,B_j_indices = compute_B_j_factors(Y,params,extras);
  K_j_list,K_j_indices = compute_bar_K_j(Y,params,extras);
  
  num_heat_bodies = params['num_heat_bodies']; # params['num_heat_bodies'];

  k_B = params['k_B'];
  for j in range(0,num_heat_bodies):
    B_j = B_j_list[j];
    K_j = K_j_list[j];
    B_j_B_j_T = np.matmul(B_j,B_j.T); 
    N_j = B_j_B_j_T/(2.0*k_B); 

    nn = Y.shape[0];
    M_j = np.zeros((nn,nn));
    I_out = I_in = B_j_indices['I_out'][j]; # uses BB^T is square
    I_local_out = I_local_in = B_j_indices['I_local_out'][j]; # uses BB^T is square
    add_in_matrix_entries(M_j,N_j,I_in,I_out,I_local_in,I_local_out);
  
    nn = Y.shape[0];
    KK_j = np.zeros((nn,nn));
    I_out = K_j_indices['I_out'][j]; 
    I_in = K_j_indices['I_in'][j]; 
    I_local_out = K_j_indices['I_local_out'][j]; 
    I_local_in = K_j_indices['I_local_in'][j]; 
    add_in_matrix_entries(KK_j,K_j,I_in,I_out,I_local_in,I_local_out);

    err_w_K = np.abs(M_j - KK_j).max();
    print("j = " + str(j));
    print("err_w_K = %.2g"%err_w_K); 
    print("");

def test_div_K_j(Y,params,j,I0,epsilon=None,extras=None):

  if epsilon is None:
    epsilon = 1e-5;

  # perturb the current state
  Yp = 0*Y; Yp[I0] = Y[I0] + epsilon; 

  # compute K_j
  K_j_list, K_j_indices = compute_bar_K_j(Y,params,extras);
  K_j_p_list, K_j_p_indices = compute_bar_K_j(Yp,params,extras);

  # call function for div_K_j 
  div_K_j_list = compute_div_K_j(Y,params,extras);
  div_K_j = div_K_j_list[j]; 

  # compute estimate
  div_est = (K_j_p_list[j][I0] - K_j_list[j][I0])/epsilon;
  div_comp = div_K_j_list[j][I0];
  
  # compute error 
  err_div = np.abs(div_comp - div_est);

  # report results 
  print("test divergence (epsilon = %.2e):"%epsilon);
  print("j = %d, I0 = %d"%(j,I0));
  print("div_est = %.2e"%div_est);
  print("div_comp = %.2e"%div_comp);
  print("err_div = %.2e"%err_div);
  print("");

def test_div_K_j_set0(Y,params,extras=None):
  Y_I = params['Y_I'];

  epsilon = 1e-5;
  
  # perform a bunch of div_K_j tests 

  # fluid
  j = 1;

  for k in range(0,4):
    I0 = Y_I['I1_fluid_p'] + k;
    test_div_K_j(Y,params,j,I0,epsilon,extras); 

  # interface
  j = 2;

  for k in range(0,4):
    I0 = Y_I['I1_fluid_p'] + k;
    test_div_K_j(Y,params,j,I0,epsilon,extras); 

def test_div_K_j_mc(Y,params,extras=None):
  """ Monte-Carlo estimate of div_K_j and testing errors """
  num_heat_bodies = params['num_heat_bodies'];
  print("--");
  print("estimating div_K_j using monte-carlo methods");
  params_mc = {'num_samples':int(1e4),'delta':1e-2,'flag_verbose':2};
  params_mc.update({'flag_save_file':True,'save_skip':params_mc['num_samples']//10});
  print("params_mc = " + str(params_mc));
  K_j_list, K_j_indices = compute_bar_K_j(Y,params);
  B_j_list,B_j_indices = compute_B_j_factors(Y,params);
  div_K_j_list = compute_div_K_j(Y,params);
  div_K_j_mc_list = compute_div_K_j_monte_carlo(Y,params,**params_mc);
  ss = {'div_K_j_mc_list':div_K_j_mc_list,
        'params_mc':params_mc,
        'K_j_list':K_j_list,
        'K_j_indices':K_j_indices,
        'B_j_indices':B_j_indices,
        'div_K_j_list':div_K_j_list};
  print("estimates computed.");
  err_div_K_j_list = [];
  for j in range(0,num_heat_bodies):
    err_div_K_j = np.abs(div_K_j_list[j] - div_K_j_mc_list[j]).max();
    err_div_K_j_list.append(err_div_K_j);
    print("err_div_K_j[%.d] = %.2e"%(j,err_div_K_j));
  ss.update({'err_div_K_j_list':err_div_K_j_list});
  filename = base_dir + '/debug/' + 'test_div_K_j_mc.pickle';
  print("filename = " + filename);
  fid = open(filename,"wb"); pickle.dump(ss,fid); fid.close();
  print("--");   
  print("");



# =======================================
# conc bar_K_j cases
# ---------------------------------------
def test_B_j__conc(Y,params,extras=None):
  if extras is None:
    extras = {};

  extras_bar_K_j, = tuple(map(extras.get,['extras_bar_K_j']));

  B_j_list,B_j_indices = compute_B_j_factors__conc(Y,params,extras);
  K_j_list,K_j_indices = compute_bar_K_j_conc(Y,params,extras_bar_K_j);
  
  num_heat_bodies = params['num_heat_bodies']; # params['num_heat_bodies'];

  k_B = params['k_B'];
  for j in range(0,num_heat_bodies):
    B_j = B_j_list[j];
    K_j = K_j_list[j];
    B_j_B_j_T = np.matmul(B_j,B_j.T); 
    N_j = B_j_B_j_T/(2.0*k_B); 

    nn = Y.shape[0];
    M_j = np.zeros((nn,nn));
    I_out = I_in = B_j_indices['I_out'][j]; # uses BB^T is square
    I_local_out = I_local_in = B_j_indices['I_local_out'][j]; # uses BB^T is square
    add_in_matrix_entries(M_j,N_j,I_in,I_out,I_local_in,I_local_out);
  
    nn = Y.shape[0];
    KK_j = np.zeros((nn,nn));
    I_out = K_j_indices['I_out'][j]; 
    I_in = K_j_indices['I_in'][j]; 
    I_local_out = K_j_indices['I_local_out'][j]; 
    I_local_in = K_j_indices['I_local_in'][j]; 
    add_in_matrix_entries(KK_j,K_j,I_in,I_out,I_local_in,I_local_out);

    err_w_K = np.abs(M_j - KK_j).max();
    print("j = " + str(j));
    print("err_w_K = %.2g"%err_w_K); 
    print("");

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
  #conc_theta = get_comp(Y,'I1_conc_theta','I2_conc_theta',Y_I);

  if extras is not None:
    G, = tuple(map(extras.get,['G']));
  else:
    G = (None);

  if G is None:
    G = compute_matrix_vec_grad(Y,params);

  # local transpose the individual blocks of the tensor
  # split into tensor with indexing (mesh_I,tensor_i,tensor_j,mesh_J,vec_i)

  # @@@ Change below for K_ovdc case for concentration
  GG = G.reshape(num_mesh_pts,num_dim,num_mesh_pts);
  c0 = c0_conc;
  factor = np.expand_dims(qr*bar_kappa/(c0*deltaV),(1,2)); # @@@ deltaV since D_S_j has deltaV
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
  #dot_G = c0*np.expand_dims(grad_r_phi.flatten(),1); # @@@ (need use G operator for perfect symmetry)  
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
  """ K_conc_heat based on a finite-volume-like model using 
  local transfers similar to Fourier's Law formulated in
  terms of 1/theta.  Gives final action similar to 
  central difference Laplacian in terms of temperature.
  Operator itself operates on 1/theta and is positive 
  semi-definite.  """
    
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

  #K_heat = K_heat/deltaV; # scaling 1/deltaV for density 
 
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
  # modification of the fluid case 
  # note subtle issues implementing correctly the operator, see $\square$ placement in notes.
  j = c_v_I['conc'];  # j = 4  
  #if flag_save: D_j_list.append(None);
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
  # Hence, $\square$ notation in our notes for this subtle ordering of
  # operations in this linear operator.  Expand first over the theta columns
  # appropriately since vector quantities then apply the divergence operator,
  # so would happen in the correct order (as in notes discussion).  The
  # operator is given by action of  $\frac{\mbox{\small div}\left(K_{visco}
  # \square \dot{\mb{F}} \right)}{\partial_\tau u}$
  aa = K_ovdc_dot_G;  # action operator multiplication by 1/theta_C(x_m) at each x-location 
  #bb = np.zeros((num_mesh_pts,num_dim,num_mesh_pts)); # num_mesh_pts = num_theta 
  #aaa = np.expand_dims(aa,1);
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
#    bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta] += np.diag(dot_G[d::num_dim_sq]*K_ovdc_dot_G[d::num_dim_sq]);
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

# =================================
# particle (overdamped case)
# ---------------------------------
def compute_D_j_inv_particle(Y,params,extras=None):
  gamma_particle,Y_I,c_v,c_v_I = tuple(map(params.get,['gamma_particle','Y_I','c_v','c_v_I']));
  num_particles,num_dim = tuple(map(params.get,['num_particles','num_dim']));

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  interface_q,interface_p,interface_theta = tuple(map(dd.get,['interface_q','interface_p','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  # locals().update(**Y_I)  # quick way to make dict elements local variables (not best practice though)
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  num_q = num_particle_q;
  #D_j = np.zeros((num_p, num_p));   
  D_j_inv = (1.0/gamma_particle)*np.eye(num_q);  

  return D_j_inv;

def compute_K_ovd_particle(Y,params,extras=None):
  Y_I,c_v,c_v_I = tuple(map(params.get,['Y_I','c_v','c_v_I']));
  num_particles,num_dim = tuple(map(params.get,['num_particles','num_dim']));

  get_parts = params['func_get_parts'];

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));

  Y_I_parts = dd = get_parts_I(params);
  # locals().update(**Y_I)  # quick way to make dict elements local variables (not best practice though)
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
  # modification of the fluid case 
  # note subtle issues implementing correctly the operator, see $\square$ placement in notes.
  j = c_v_I['particle'];  # j = 4  
  #if flag_save: D_j_list.append(None);
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
  #D_j_inv = compute_D_j_inv_particle(Y,params); # compute the inverse D^{-1} tensor
  #theta_P__D_inv = theta_P*D_j_inv;
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
  """
  Dissipation at the conc-structure interface.  Note, one would need one 
  temperature for each distinct micro-structure.  This would then correspond to 
  having a dissipative operator for each of the micro-structures. 
  """
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
  # locals().update(**Y_I)  # quick way to make dict elements local variables (not best practice though)
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  Y_I_parts = dd = get_parts_I(params);
  I1_conc_q,I2_conc_q = tuple(map(dd.get,['I1_conc_q','I2_conc_q']));
  I1_conc_theta,I2_conc_theta = tuple(map(dd.get,['I1_conc_theta','I2_conc_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  num_q = num_particle_q + num_conc_q;
  #D_j = np.zeros((num_p, num_p)); 
  D_j = np.zeros((num_q, num_q));  # assumes only one temperature to track accumulated dissipation energy
 
  # particle drag term from the interface 
  matrix_Gamma_op = compute_matrix_Gamma_op(Y,params); 

  # conc drag term from the interface 
  #D_j[ii1_conc_q:ii2_conc_q,ii1_conc_q:ii2_conc_q] = 0.0; # @@@ double-check, if need this.. = gamma*np.dot(matrix_Gamma_op.T,matrix_Gamma_op)/(deltaV*deltaV); # factor deltax_sq given energy density for conc # @@@ check 

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
    energy_flux_list = [];  # one for each irreversible process 

  Y_parts = dd = get_parts(Y,params);
  conc_q, conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  particle_q, particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  Y_I_parts = dd = get_parts_I(params);
  # locals().update(**Y_I)  # quick way to make dict elements local variables (not best practice though)
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
  #if flag_save: D_j_list.append(D_j);

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
 
  #extras_D_j_interface = {'flag_save':True};
  #D_j = compute_D_j_interface(Y,params,extras_D_j_interface); ii = c_v_I;
  #if flag_save: D_j_list.append(D_j);
  ii = c_v_I;
  d_tau_u = C_I = partial_theta_I_u_j = partial_theta_j_u_j = c_v[ii['interface']];
  
  # interface friction terms 
  #theta_I = interface_theta; 
  #bar_K_j[I1_p:I2_p,I1_p:I2_p] = theta_I*D_j; 
  #bar_K_j[I1_p:I2_p,I1_theta3:I2_theta3] = np.expand_dims(-theta_I*np.dot(D_j,dot_q_fld_dx)/partial_theta_I_u_j,1); 
  #bar_K_j[I1_theta3:I2_theta3,I1_p:I2_p] = np.transpose(bar_K_j[I1_p:I2_p,I1_theta3:I2_theta3]);
  ##bar_K_j[j,I1_theta3:I2_theta3,I1_theta3:I2_theta3] = theta_I*np.dot(dot_q,np.dot(D_j,dot_q))/(partial_theta_I_u_j*partial_theta_I_u_j);
  #bar_K_j[I1_theta3:I2_theta3,I1_theta3:I2_theta3] = np.dot(dot_q_fld_dx,-1.0*bar_K_j[I1_p:I2_p,I1_theta3:I2_theta3])/partial_theta_I_u_j;

  # heat exchnage terms between the interface, conc, and particle(s)   
  ii = c_v_I;
  c_P = cc_P = c_v[ii['particle']]; c_C = cc_C = c_v[ii['conc']]; 
  c_I = cc_I = c_v[ii['interface']]; 
  c_I_inv_dx = cc_I_inv_dx = cc_I/deltaV; # arises since specific heat of c_I per unit volume needed for conc exchange
  c_C_inv_dx = cc_C_inv_dx = cc_C/deltaV; # arises since specific heat of c_C per unit volume needed for conc exchange
  c_I_I = partial_tau_u_I_I = cc_I*cc_I; c_I_I_inv_dx = cc_I*cc_I_inv_dx;
  c_C_C = partial_tau_u_C_C = cc_C*cc_C; c_C_C_inv_dx = cc_C*cc_C_inv_dx;
  c_P_P = partial_tau_u_P_P = cc_P*cc_P; c_P_I = partial_tau_u_P_I = cc_P*cc_I; 
  c_C_I = partial_tau_u_C_I = cc_C*cc_I; c_C_I_inv_dx = cc_C*cc_I_inv_dx;
  theta_P = particle_theta; theta_C = conc_theta; theta_I = interface_theta; 
  Gamma_op_vec = matrix_Gamma_op = extras_D_j_interface['matrix_Gamma_op'];
  aa = Gamma_op_vec.reshape(num_particles,num_dim,num_mesh_pts,num_dim);
  Gamma_op = Gamma_op_scalar = aa[:,0,:,0].reshape(num_particles,num_mesh_pts); # get scalar operator
  Lambda_op = Gamma_op.T;
  #kappa_C_I_dx = kappa_C_I*Lambda_op.flatten();  # spatial dependence of thermal conductivity
  kappa_C_I_xx_dx = kappa_C_I*Lambda_op.flatten();  # spatial dependence of thermal conductivity
  kappa_C_I_xx = kappa_C_I_xx_dx/deltaV;  # spatial dependence of thermal conductivity (so integrates)
  # these scalings follow, since we want conservation of internal energies in heat exchanges 
  # E[theta_C,theta_I] = sum_m c_C*theta_C(x_m) deltaV + c_I*theta_I.
  
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
    rate_heat_particle = 0.0; # @@@#np.sum(np.dot(bar_K_j[I1_theta:I2_theta,I1_theta:I2_theta],(c_P/theta_P)))*c_P;
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
 
  #R_ovd = np.zeros((num_particle_q,num_particle_q)); 
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
 
  if extras is not None and 'dot_G' in extras: # warning: invalidate, so do not forget to update each time 
    extras.update({'dot_G':None}); 

  if flag_save: 
    extras.update({'matrix_vec_div':matrix_vec_div}); 
    
  return R_ovdc; 


def compute_R_heat2__conc(Y,params,extras):

  """ 
    Compute the factor $K_heat = RR^T$.

    This uses the discrete operator based on finite-volume-like method
    and boxes.   This yields the factorization of the form 
  
    R = -D, where 
    $[G (1/\theta)]_{(I_1 + I_2)/2} = s_{I_1,I_2}\sqrt{\theta_{I_1}\theta_{I_2}}\left(1/\theta_{I_2} - 1/\theta_{I_1}
    \right)$, where $s_{I_1,I_2} = -1$ is $I_1 < I_2$ and $+1$ otherwise.  We claim
     that in fact we have $K_{heat} = -D\cdot G$ gives the operator. 

  """
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

  # we use staggered convention that I0^(d) is always 
  # to the left or below the cell-center I0^{d).
  # we construct the weighted gradient operator (wG) 
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
  #B_j = sqrt_two_k_B*np.zeros((n1,n2));
  B_j = np.zeros((n1,n2));

  # compute the entries
  # R_D_inv = gamma_particle*np.eye(num_particle_q);
  # K_ovd = compute_K_ovd_particle(Y,params,extras); 
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
 
#  if bar_K_j is None:
#    bar_K_j,bar_K_j_indices = \
#      compute_bar_K_j__conc(Y,params,extras);
#    if extras is not None:
#      extras_matrix_vec_div = \
#        extras['extras_matrix_vec_div'];
#      extras_K_ovdc_dot_G = extras['extras_K_ovdc_dot_G'];

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
 
#  if bar_K_j is None:
#    bar_K_j,bar_K_j_indices = \
#      compute_bar_K_j__conc(Y,params,extras);
#    if extras is not None:
#      extras_matrix_vec_div = \
#        extras['extras_matrix_vec_div'];
#      extras_K_ovdc_dot_G = extras['extras_K_ovdc_dot_G'];

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
  # The interface here only serves to exchange heat between the particle and conc fields.
  # In previous work, we put some of the particle exchanges into the interface tensor.
  # This just involves the three temperature fields. 
  #
  # we break down the generation of fluctuations into parts 
  # so that g = R_1 x_1 + R_2 x_2.  This can be combined into
  # one large B matrix of the form B=[R_1|R_2] for xi = [xi_1|x_2]^T.
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

  R_21[0,0] = np.sqrt(kappa_P_I*theta_P[0]*theta_I[0])*(1.0/c_P); # WARNING: assume 1 particle  
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

  #R_22_x = R_22_x/deltaV; # @@@ deltaV scaling suggested by numerical studies 

  # @@@ check indexing below
  # copy into B_j 
  i1 = i1_B_theta_C; i2 = i1 + num_conc_theta;
  ii_range = range(i1,i2); 
  j1 = j1_B_theta_C; j2 = j1 + num_conc_theta;
  jj_range = range(j1,j2);
  B_j[ii_range,jj_range] += R_22_x[i1_R22_theta_C,:];

  # @@@ check indexing below
  #i1 = i1_B_theta_I; i2 = i1 + num_interface_theta;
  ii_range = i1_B_theta_I*np.ones(num_conc_theta,dtype=int);
  j1 = j1_B_theta_C; j2 = j1 + num_conc_theta;
  jj_range = range(j1,j2);
  B_j[ii_range,jj_range] += R_22_x[i1_R22_theta_I,:];  
 
  I_in = None; # signal not relevant since random noise generation 
  I_local_in = None; # signal not relevant since random noise generation 
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
  """
  Uses factorization of $bar_K_j = N_E K_0 N_E^*$ and 
  $bar_K_j = M_E K_0 M_E^*$ to compute the needed 
  factors for the noise generation $RR^T = K$ 
  """
  
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
  #
  # For a summary, see pg. 26 of 
  # H. Ottinger book, Beyond Eqilibrium.
  #
  # We combine this with our factorizations
  # and other results. 
  #
  # g_thm dt =  k_B*div_Y(K)dt + B(Y)*dW_t,
  # where, 
  # B(Y)B(Y)^T = 2*k_B K(Y).
  #

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

