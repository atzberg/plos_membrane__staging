# protein-membrane simulations with fluctuating concentration fields 

print("="*80);

# -- imports
import numpy as np
import pickle;
import time; 
import os; 
import sys;
import shutil;
import re; 

import argparse; 

import pkg.selm_ne as sne; 

script_base_name = 'conc_field_01';
script_ext = 'py';

# --
# functions

def create_Y_I(params):
  num_mesh_x,num_mesh_y,num_particles,num_dim =\
    tuple(map(params.get,['num_mesh_x','num_mesh_y',
                          'num_particles','num_dim']));
  Y_I = {};
  num_interfaces = num_particles; I = 0; 

  Y_I['I1_particle_q'] = I; I += num_particles*num_dim; Y_I['I2_particle_q'] = I;
  Y_I['I1_conc_q'] = I; I += num_mesh_x*num_mesh_y; Y_I['I2_conc_q'] = I;
  Y_I['I1_interface_q'] = I; I += num_interfaces*num_dim; Y_I['I2_interface_q'] = I;

  Y_I['I1_particle_theta'] = I; I += num_particles; Y_I['I2_particle_theta'] = I;
  Y_I['I1_conc_theta'] = I; I += num_mesh_x*num_mesh_y; Y_I['I2_conc_theta'] = I;
  Y_I['I1_interface_theta'] = I; I += num_interfaces; Y_I['I2_interface_theta'] = I;

  return Y_I; 

def get_parts(Y,params):
  dd = {};
  dd['Y_I'] = Y_I = params['Y_I'];

  dd['particle_q'] = sne.get_comp(Y,'I1_particle_q','I2_particle_q',Y_I);
  dd['particle_theta'] = sne.get_comp(Y,'I1_particle_theta','I2_particle_theta',Y_I);

  dd['conc_q'] = sne.get_comp(Y,'I1_conc_q','I2_conc_q',Y_I);
  dd['conc_theta'] = sne.get_comp(Y,'I1_conc_theta','I2_conc_theta',Y_I);

  dd['interface_q'] = sne.get_comp(Y,'I1_interface_q','I2_interface_q',Y_I);
  dd['interface_theta'] = sne.get_comp(Y,'I1_interface_theta','I2_interface_theta',Y_I);

  return dd;

def phi_energy__eta_gauss_01(Y,extras):
  params,k1,sigma_sq,flag_compute_grad = \
    tuple(map(extras.get,['params','k1','sigma_sq',
                          'flag_compute_grad']));

  if flag_compute_grad is None:
    flag_compute_grad = True;

  dd = get_parts(Y,params);

  x1,x2,Lx,Ly = sne.get_mesh_coord(params); # locations of lattice sites
  xx = np.vstack((x1,x2)).T;

  particle_q = dd['particle_q']; conc_q = dd['conc_q'];
  X = np.array([particle_q[0],particle_q[1]]);
  XX = np.expand_dims(X,0); num_dim = XX.shape[1];

  eta = np.exp(-np.sum(np.power((xx - XX),2),1)/(2.0*sigma_sq)); 
  ZZ = np.power(np.sqrt(2*np.pi*sigma_sq),num_dim);
  eta = k1*eta/ZZ;

  # energy
  phi = -eta; # potential negative, so attracted 

  # gradient
  if flag_compute_grad:
    grad_r_eta = -((xx - XX)/sigma_sq)*np.expand_dims(eta,1); 
    grad_r_phi = -grad_r_eta; # note energy -eta above  
    grad_X_phi = grad_r_eta;  
  else:
    grad_r_phi = None;
    grad_X_phi = None; 

  return phi,grad_r_phi,grad_X_phi; 

def phi_energy__zero(Y,extras):
  params,k1,sigma_sq,flag_compute_grad = \
    tuple(map(extras.get,['params','k1','sigma_sq',
                          'flag_compute_grad']));

  if flag_compute_grad is None:
    flag_compute_grad = True;

  dd = get_parts(Y,params);

  x1,x2,Lx,Ly = sne.get_mesh_coord(params); # locations of lattice sites
  xx = np.vstack((x1,x2)).T;

  particle_q = dd['particle_q']; conc_q = dd['conc_q'];
  X = np.array([particle_q[0],particle_q[1]]);
  XX = np.expand_dims(X,0); num_dim = XX.shape[1];

  # energy
  phi = 0*xx[:,0]; 

  # gradient
  if flag_compute_grad:
    grad_r_eta = 0*xx;
    grad_r_phi = -grad_r_eta; # note energy -eta above  
    grad_X_phi = grad_r_eta;  
  else:
    grad_r_phi = None;
    grad_X_phi = None; 

  return phi,grad_r_phi,grad_X_phi; 

def phi_energy__sin_01(Y,extras):
  params,k0 = tuple(map(extras.get,['params','k0']));
  dd = get_parts(Y,params);
  x1,x2,Lx,Ly = sne.get_mesh_coord(params); # locations of lattice sites
  conc_q = dd['conc_q']; particle_q = dd['particle_q']; 

  X = np.array([particle_q[0],particle_q[1]]);
  grad_X_phi = 0*X; 

  phi = np.sin(k0*x1);
  grad_r_phi = np.vstack((k0*np.cos(k0*x1),0*x1)).T;

  return phi,grad_r_phi,grad_X_phi; 

def U_q_energy__coupling_conc(Y,extras):
  params, = tuple(map(extras.get,['params']));
  c0_conc,deltaX,func_phi,extras_func_phi \
          = tuple(map(params.get,
                  ['c0_conc','deltaX',
                   'func_phi','extras_func_phi']));

  deltaV = deltaX_sq = deltaX*deltaX; 
  c0 = c0_conc;

  if func_phi is None: # resolve the function ref 
    func_phi_str, = tuple(map(extras.get,['func_phi_str']));
    cmd_str = "extras['func_phi'] = %s"%func_phi_str; # map str to function reference 
    exec(cmd_str); 
    func_phi = extras['func_phi'];

  phi,grad_r_phi,grad_X_phi = func_phi(Y,extras_func_phi);
  
  dd = get_parts(Y,params);
  q = conc_q = dd['conc_q']; 
 
  # compute the integral and derivative in X 
  U_q = np.sum(phi*c0*q)*deltaV;
  grad_X_U_q = np.sum(grad_X_phi*c0*np.expand_dims(q,1),0)*deltaV;

  return U_q,grad_X_U_q;

def U_q_energy__sin_01(Y,extras):
  params,k0 = tuple(map(extras.get,['params','k0']));
  dd = get_parts(Y,params);
  x1,x2,Lx,Ly = sne.get_mesh_coord(params); # locations of lattice sites
  particle_q = dd['particle_q'];
  
  x1 = particle_q[0]; x2 = particle_q[1];
  U_q = np.sin(k0*x1); # first component of particle
  grad_U_q = np.vstack((k0*np.cos(k0*x1),0*x1)).T;

  return U_q,grad_U_q; 

def func_Psi__well_01(XX,extras):
  sigma_sq,c2,X_0 = tuple(map(extras.get,['sigma_sq','c2','X_0']));

  XX_0 = np.expand_dims(X_0,0);
  Psi = -c2*np.exp(-np.sum(np.power((XX - XX_0),2),1)/(2.0*sigma_sq)); 
  grad_Psi = -((XX - XX_0)/sigma_sq)*np.expand_dims(Psi,1); 

  return Psi,grad_Psi; 

def func_Psi__well_02(XX,extras):
  sigma_sq,c2,X_0 = tuple(map(extras.get,['sigma_sq','c2','X_0']));

  XX_0 = np.expand_dims(X_0,0);

  x0 = 5/3.0; y0 = 5/3.0;
  num_x1 = 4; num_x2 = 4;
  sx = 0; sy = 0; 
  dx = 2.0/(num_x1-1); dy = 2.0/(num_x2-1); 

  Psi = np.zeros(XX.shape[0]); 
  grad_Psi = 0*XX;
  for k1 in range(-1,num_x1):
    for k2 in range(-1,num_x2):
      if np.mod(k2 + 1,2) == 0:
        sx = 0.0;
      else:
        sx = 0.0;
      if np.mod(k1 + 1,2) == 0:
        sy = dy/2.0; 
      else:
        sy = 0.0; 
      xp0 = x0 - k1*dx - sx; yp0 = y0 - k2*dy - sy;
      XX_0[0,0] = xp0; XX_0[0,1] = yp0; 
      Psi_single = -c2*np.exp(-np.sum(np.power((XX - XX_0),2),1)/(2.0*sigma_sq)); 
      Psi += Psi_single;
      grad_Psi += -((XX - XX_0)/sigma_sq)*np.expand_dims(Psi_single,1); 

  return Psi,grad_Psi; 

def func_Psi__zero(XX,extras):
  Psi = 0.0; grad_Psi = 0*XX;

  return Psi,grad_Psi; 

def func_eta__gauss_01(xx,X,extras):
  sigma_sq, = tuple(map(extras.get,['sigma_sq']));

  XX = np.expand_dims(X,0); num_dim = XX.shape[1];
  eta = np.exp(-np.sum(np.power((xx - XX),2),1)/(2.0*sigma_sq)); 
  ZZ = np.power(np.sqrt(2*np.pi*sigma_sq),num_dim);
  eta = eta/ZZ;
  grad_eta = -((xx - XX)/sigma_sq)*np.expand_dims(eta,1); 

  return eta,grad_eta; 

def U_q_energy__conc_grad_01(Y,extras):
  func_Psi,extras_Psi = tuple(map(extras.get,['func_Psi','extras_Psi'])); 

  # -- conc coupling energy
  U_q__conc,grad_X_U_q__conc = U_q_energy__coupling_conc(Y,extras);

  # -- Psi(X) energy 
  if func_Psi is None:
    func_Psi_str, = tuple(map(extras.get,['func_Psi_str'])); 
    if func_Psi_str is not None:
      cmd_str = "extras['func_Psi'] = %s"%func_Psi_str;
      exec(cmd_str); 
      func_Psi = extras['func_Psi'];

  #if funcPsi is not None:
  dd = get_parts(Y,params);
  particle_q = dd['particle_q']; 
  X = np.array([particle_q[0],particle_q[1]]);
  XX = np.expand_dims(X,0);

  U_q__psi,grad_X_U_q__psi = func_Psi(XX,extras_Psi);

  # -- total
  U_q = U_q__conc + U_q__psi; 
  grad_X_U_q = grad_X_U_q__conc + grad_X_U_q__psi;

  return U_q,grad_X_U_q; 

def compute_S_j(Y,params):
  Y_I,c_v,c_v_I = tuple(map(params.get,['Y_I','c_v','c_v_I']));
  num_particles,num_dim,deltaX = tuple(map(
    params.get,['num_particles','num_dim','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX; 
  c0_conc, = tuple(map(params.get,['c0_conc']));

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];
 
  s_j = np.zeros(num_particle_theta + num_conc_theta + num_interface_theta);
  ii = c_v_I; 

  j = ii['particle']; c_P = c_v[j];
  i1 = 0; i2 = i1 + num_particle_theta; 
  c_j_of_q = 0; 
  s_j[i1:i2] = c_P*np.log(particle_theta) + c_j_of_q;

  j = ii['conc']; c_C = c_v[j];
  i1 = i2; i2 = i1 + num_conc_theta; 
  c0 = c0_conc;
  s_j[i1:i2] = -c0*conc_q*np.log(conc_q); # entropy density contribution
  s_j[i1:i2] += c_C*np.log(conc_theta);

  j = ii['interface']; c_I = c_v[j];
  i1 = i2; i2 = i1 + num_interface_theta; 
  c_j_of_q = 0; 
  s_j[i1:i2] = c_I*np.log(interface_theta) + c_j_of_q;

  return s_j;  

def compute_S(Y,params,extras=None):
  Y_I,c_v,c_v_I = tuple(map(params.get,['Y_I','c_v','c_v_I']));
  num_particles,num_dim,deltaX = tuple(map(
    params.get,['num_particles','num_dim','deltaX']));

  deltaV = deltaX_sq = deltaX*deltaX; 

  if extras is not None:
    flag_save = extras['flag_save'];
  else:
    flag_save = None; 

  if flag_save is None:
    flag_save = False; 

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = \
    tuple(map(dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = \
    tuple(map(dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = \
    tuple(map(dd.get,['interface_q','interface_theta']));

  I1_particle_q,I2_particle_q = \
    tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = \
    tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; 
  num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; 
  num_interface_theta = interface_theta.shape[0];

  # compute entropies
  entropy = 0; ii = c_v_I; 
  S_j = compute_S_j(Y,params); 

  j = ii['particle'];
  i1 = 0; i2 = i1 + num_particle_theta; 
  entropy += np.sum(S_j[i1:i2]); 

  j = ii['conc'];
  i1 = i2; i2 = i1 + num_conc_theta; # same as num_conc_q  
  entropy += np.sum(S_j[i1:i2])*deltaV; # conc s_j(x_m) is an entropy density 

  j = ii['interface'];
  i1 = i2; i2 = i1 + num_interface_theta; 
  entropy += np.sum(S_j[i1:i2]);

  if flag_save:
    extras['S_j'] = S_j;  

  return entropy;

def compute_U_j(Y,params):
  Y_I,c_v,c_v_I = tuple(map(params.get,['Y_I','c_v','c_v_I']));

  num_particles,num_dim,deltaX =\
    tuple(map(params.get,['num_particles','num_dim','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX;

  func_phi,extras_func_phi = tuple(map(params.get,
                                       ['func_phi','extras_func_phi']));

  func_U_q,extras_func_U_q = tuple(map(params.get,
                                       ['func_U_q','extras_func_U_q']));

  c0_conc, = tuple(map(params.get,['c0_conc']));

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];
 
  u_j = np.zeros(num_particle_theta + num_conc_theta + num_interface_theta);
  ii = c_v_I; 

  j = ii['particle']; c_P = c_v[j];
  i1 = 0; i2 = i1 + num_particle_theta; 
  U_q,grad_U_q = func_U_q(Y,extras_func_U_q);
  u_j[i1:i2] = U_q; 
  c_j_of_q = 0; 
  u_j[i1:i2] = c_P*particle_theta + c_j_of_q;

  j = ii['conc']; c_C = c_v[j]; 
  i1 = i2; i2 = i1 + num_conc_theta; 
  phi,grad_r_phi,grad_X_phi = func_phi(Y,extras_func_phi);
  u_j[i1:i2] = c0_conc*phi*conc_q; 
  c_j_of_q = 0; 
  u_j[i1:i2] += c_C*conc_theta + c_j_of_q; 

  j = ii['interface']; c_I = c_v[j];
  i1 = i2; i2 = i1 + num_interface_theta; 
  c_j_of_q = 0; 
  u_j[i1:i2] = c_I*interface_theta + c_j_of_q;

  return u_j;  

def compute_E_kinetic_j(Y,params): 
  Y_I,c_v = tuple(map(params.get,['Y_I','c_v']));
  deltaX,num_dim = tuple(map(params.get,['deltaX','num_dim'])); 
  deltaV = deltaX_sq = deltaX*deltaX;

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  E_j = np.zeros(num_particle_theta + num_conc_theta + num_interface_theta);

  # kinetic energy of the particles 
  #E_j[i1:i2] = 0.0;

  # kinetic energy of the interface 
  #E_j[i1:i2] = 0.0;

  return E_j; 

def compute_E(Y,params,extras=None):
  Y_I,c_v,c_v_I = tuple(map(params.get,['Y_I','c_v','c_v_I']));

  num_particles,num_dim,deltaX = tuple(map(
    params.get,['num_particles','num_dim','deltaX']));
  deltaV = deltaX_sq = deltaX*deltaX; 

  if extras is not None:
    flag_save,flag_parts = tuple(map(extras.get,['flag_save','flag_parts']));
  else:
    flag_save,flag_parts = None,None;

  if flag_save is None:
    flag_save = False; 

  if flag_parts is not None: 
    flag_compute_particle,flag_compute_conc,flag_compute_interface = \
      tuple(map(flag_parts.get,['particle',
        'conc','interface'])); 
  else:
    flag_compute_particle = True;
    flag_compute_conc = True;
    flag_compute_interface = True;

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  # kinetic energy 
  energy = 0; ii = c_v_I; 
  E_j = compute_E_kinetic_j(Y,params); 

  i1 = 0; i2 = i1 + num_particle_theta;
  energy += np.sum(E_j[i1:i2]); 

  i1 = i2; i2 = i1 + num_conc_theta
  energy += np.sum(E_j[i1:i2])*deltaV; # conc energy density E_j

  i1 = i2; i2 = i1 + num_interface_theta;
  energy += np.sum(E_j[i1:i2]);

  if flag_save:
    extras['E_kinetic_j'] = E_j;  

  # internal energy of the particles, conc, and interface
  U_j = compute_U_j(Y,params);

  i1 = 0; i2 = i1 + num_particle_theta;
  energy += np.sum(U_j[i1:i2]); 

  i1 = i2; i2 = i1 + num_conc_theta;
  energy += np.sum(U_j[i1:i2])*deltaV; # conc energy density U_j 

  i1 = i2; i2 = i1 + num_interface_theta;
  energy += np.sum(U_j[i1:i2]);

  if flag_save:
    extras['U_j'] = U_j;  

  return energy;

def compute_D_E(Y,params,extras=None):

  if extras is not None:
    flag_save,flag_compute_parts \
      = tuple(map(extras.get,['flag_save',
                              'flag_compute_parts']));
  else:
    flag_save = None;
    flag_compute_parts = None; 

  if flag_compute_parts is not None: 
    flag_compute_particle,flag_compute_conc,flag_compute_interface = \
      tuple(map(flag_compute_parts.get,['particle',
        'conc','interface'])); 
  else:
    flag_compute_particle = True;
    flag_compute_conc = True;
    flag_compute_interface = True;

  Y_I,c_v,c_v_I = tuple(map(
    params.get,['Y_I','c_v','c_v_I']));
  num_particles,num_dim,deltaX = tuple(map(
    params.get,['num_particles','num_dim','deltaX']));
  num_mesh_x = params['num_mesh_x']; num_mesh_y = params['num_mesh_y']; 
  deltaV = deltaX_sq = deltaX*deltaX; 

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(
    dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(
    dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = tuple(map(
    dd.get,['interface_q','interface_theta']));

  Y_I_parts = dd = sne.get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(
    dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(
    dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0];
  num_particle_theta = particle_theta.shape[0]; 
  num_conc_q = conc_q.shape[0];  
  num_conc_theta = conc_theta.shape[0]; 
  num_interface_q = interface_q.shape[0]; 
  num_interface_theta = interface_theta.shape[0];

  # set the components of the gradient 
  ii = c_v_I;
  grad = 0*Y;

  if flag_compute_particle:
    jj = ii['particle']; 
    grad_particle_q = np.zeros(num_particle_q);
    grad_particle_theta = np.ones(num_particle_theta)*c_v[jj];

    func_U_q,extras_func_U_q = tuple(map(params.get,
                                         ['func_U_q','extras_func_U_q']));
    U_q,grad_U_q = func_U_q(Y,extras_func_U_q);

    grad_particle_q[:] = grad_U_q; 

    sne.set_comp(grad,'I1_particle_q','I2_particle_q',Y_I,grad_particle_q);
    sne.set_comp(grad,'I1_particle_theta','I2_particle_theta',Y_I,grad_particle_theta);

  if flag_compute_conc:
    jj = ii['conc']; 

    # -- grad_conc_q
    grad_conc_q = np.zeros(num_conc_q);

    # energy is E = int c0*q(r)Phi(r) dV + int c_C*theta_C dV.
    func_phi,extras_func_phi \
      = tuple(map(params.get,['func_phi','extras_func_phi']));

    phi,grad_r_phi,grad_X_phi = func_phi(Y,extras_func_phi);

    c0_conc, = tuple(map(params.get,['c0_conc']));
    c0 = c0_conc;

    grad_conc_q[:] = c0*phi*deltaV;  

    # -- grad_conc_theta
    grad_conc_theta = np.ones(num_conc_theta)*c_v[jj]*deltaV; # need deltaV for energy gradient of box
    sne.set_comp(grad,'I1_conc_q','I2_conc_q',Y_I,grad_conc_q);
    sne.set_comp(grad,'I1_conc_theta','I2_conc_theta',Y_I,grad_conc_theta);

  if flag_compute_interface:
    jj = ii['interface'];
    grad_interface_q = np.zeros(num_interface_q);
    grad_interface_theta = np.ones(num_interface_theta)*c_v[jj];
    sne.set_comp(grad,'I1_interface_q','I2_interface_q',Y_I,grad_interface_q);
    sne.set_comp(grad,'I1_interface_theta','I2_interface_theta',Y_I,grad_interface_theta);

  return grad; 

def compute_D_S_j(Y,params):
  Y_I,c_v,c_v_I = tuple(map(params.get,
    ['Y_I','c_v','c_v_I']));
  num_particles,num_dim,deltaX = tuple(map(params.get,
    ['num_particles','num_dim','deltaX']));
  c0_conc, = tuple(map(params.get,['c0_conc']));
  num_mesh_x = params['num_mesh_x']; num_mesh_y = params['num_mesh_y']; 
  deltaV = deltaX_sq = deltaX*deltaX; 

  Y_parts = dd = get_parts(Y,params);
  particle_q,particle_theta = tuple(map(dd.get,['particle_q','particle_theta']));
  conc_q,conc_theta = tuple(map(dd.get,['conc_q','conc_theta']));
  interface_q,interface_theta = tuple(map(dd.get,['interface_q','interface_theta']));

  Y_I_parts = dd = sne.get_parts_I(params);
  I1_particle_q,I2_particle_q = tuple(map(dd.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(dd.get,['I1_particle_theta','I2_particle_theta']));

  num_particle_q = particle_q.shape[0]; num_particle_theta = particle_theta.shape[0];
  num_conc_q = conc_q.shape[0];  num_conc_theta = conc_theta.shape[0];
  num_interface_q = interface_q.shape[0]; num_interface_theta = interface_theta.shape[0];

  grad_D_S_j = [];
  ii = c_v_I;

  # set the components of the gradient 
  j = ii['particle']; c_P = c_v[j]; 
  num_q = num_particle_q; num_theta = num_particle_theta;
  i1_q = 0; i2_q = i1_q + num_q; 
  i1_theta = i2_q; i2_theta = i1_theta + num_theta; 

  grad = np.zeros((num_q + num_theta));
  grad_particle_q = np.zeros(num_particle_q);
  grad_particle_theta = c_P/particle_theta;

  grad[i1_q:i2_q] = grad_particle_q;
  grad[i1_theta:i2_theta] = grad_particle_theta;

  grad_D_S_j.append(grad);

  # set the components of the gradient 
  j = ii['conc']; c_C = c_v[j]; 
  num_q = num_conc_q;  num_theta = num_conc_theta;
  i1_q = 0; i2_q = i1_q + num_q;  
  i1_theta = i2_q; i2_theta = i1_theta + num_theta; 

  grad = np.zeros(num_q + num_theta);
  c0 = c0_conc;
  grad_conc_q = -c0*(np.log(conc_q) + 1.0)*deltaV; # need deltaV (since entropy per finite volume box)
  grad_conc_theta = (c_C*deltaV/conc_theta); # need deltaV (since entropy per finite volume box)

  grad[i1_q:i2_q] = grad_conc_q;
  grad[i1_theta:i2_theta] = grad_conc_theta;

  grad_D_S_j.append(grad);

  # interface
  j = ii['interface']; c_I = c_v[j]; 
  num_q = num_particle_q + num_conc_q; 
  num_theta = num_particle_theta + num_conc_theta + num_interface_theta;

  i1_q1 = 0; i2_q1 = i1_q1 + num_particle_q; 
  i1_q2 = i2_q1; i2_q2 = i1_q2 + num_conc_q; 
  i1_theta1 = i2_q2; i2_theta1 = i1_theta1 + num_particle_theta; # particle theta 
  i1_theta2 = i2_theta1; i2_theta2 = i1_theta2 + num_conc_theta; # conc theta 
  i1_theta3 = i2_theta2; i2_theta3 = i1_theta3 + num_interface_theta; # interface theta 

  grad_interface_q = np.zeros(num_interface_q);
  grad_interface_theta = c_I/interface_theta;

  grad = np.zeros(num_q + num_theta);
  grad[i1_q1:i2_q1] = grad_particle_q;
  grad[i1_q2:i2_q2] = grad_conc_q;
  grad[i1_theta1:i2_theta1] = grad_particle_theta;
  grad[i1_theta2:i2_theta2] = grad_conc_theta;
  grad[i1_theta3:i2_theta3] = grad_interface_theta;

  grad_D_S_j.append(grad);

  return grad_D_S_j; 

def export_vtk_particle_data(output_data):
  params,base_dir,base_dir_timestep,time_index,flag_verbose = tuple(map(output_data.get,
    ['params','base_dir','base_dir_timestep','time_index','flag_verbose']));
  filename = '%s/particle_%.8d.vtp'%(base_dir_timestep,time_index);
  if flag_verbose >= 2:
    print("filename = " + filename);
  Y_np1,Y_I = tuple(map(output_data.get,['Y_np1', 'Y_I']));
  I1_q,I2_q,I1_theta_P,I2_theta_P = tuple(map(Y_I.get,
    ['I1_particle_q','I2_particle_q',
     'I1_particle_theta','I2_particle_theta']));
  I1_theta_I,I2_theta_I = tuple(map(Y_I.get,
    ['I1_interface_theta','I2_interface_theta']));
  X = Y_np1[I1_q:I2_q]; 
  XX = sne.conv_2d_vec_3d(X,num_dim);
 
  theta_P = Y_np1[I1_theta_P:I2_theta_P];
  theta_I = Y_np1[I1_theta_I:I2_theta_I];

  points = XX;  

  field_list = [];
  field_list.append({'field_name':'theta_P','NumberOfComponents':1,
                     'field_values':theta_P});
  field_list.append({'field_name':'theta_I','NumberOfComponents':1,
                     'field_values':theta_I});
  
  sne.write_vtp_data(filename,points,field_list);  
 
def export_vtk_conc_data(output_data):
  params,base_dir,base_dir_timestep,time_index,flag_verbose = \
    tuple(map(output_data.get,
    ['params','base_dir','base_dir_timestep','time_index','flag_verbose']));
  filename = '%s/conc_%.8d.vtr'%(base_dir_timestep,time_index);
  if flag_verbose >= 2:
    print("filename = " + filename);
  Y_np1,Y_I = tuple(map(output_data.get,['Y_np1', 'Y_I']));
  num_mesh_x,num_mesh_y,rho,deltaX = tuple(map(params.get,
      ['num_mesh_x','num_mesh_y','rho','deltaX']));

  I1_q,I2_q,I1_theta,I2_theta = tuple(map(Y_I.get,
    ['I1_conc_q','I2_conc_q',
     'I1_conc_theta','I2_conc_theta']));
  conc_q = Y_np1[I1_q:I2_q]; 
  conc_theta = Y_np1[I1_theta:I2_theta]; 

  # points for making up the cell_data grid
  xx = [];
  xx.append(np.linspace(0,num_mesh_x,num_mesh_x + 1)*deltaX - 0.5*deltaX);
  xx.append(np.linspace(0,num_mesh_y,num_mesh_y + 1)*deltaX - 0.5*deltaX);
  xx.append(np.array([0]));

  field_list = []; 

  flag = True; 
  if flag: # testing x1,x2 indexing
    x1,x2,Lx,Ly = sne.get_mesh_coord(params);
    field_list.append({'field_name':'x1','NumberOfComponents':1,
                       'data_org':'cell_data','field_values':x1});
    field_list.append({'field_name':'x2','NumberOfComponents':1,
                       'data_org':'cell_data','field_values':x2});

  field_list.append({'field_name':'q','NumberOfComponents':1,
                     'data_org':'cell_data','field_values':conc_q});
  field_list.append({'field_name':'theta_C','NumberOfComponents':1,
                     'data_org':'cell_data','field_values':conc_theta});
  
  sne.write_vtr_data(filename,xx,field_list);  

def export_system_state(output_data):
  base_dir,base_dir_timestep,time_index,flag_verbose = tuple(map(output_data.get,
    ['base_dir','base_dir_timestep','time_index','flag_verbose']));
  flag_compute_K,flag_compute_div_K,flag_stochastic = tuple(map(output_data.get,
    ['flag_compute_K','flag_compute_div_K','flag_stochastic']));
  filename = '%s/system_state_%.8d.pickle'%(base_dir_timestep,time_index);
  if flag_verbose >= 2:
    print("filename = " + filename);
  extract_keys = ['Y_n','E','E_kinetic_j','U_j','S','S_j','D_E','D_S_j'];
  if flag_stochastic:
    extract_keys += ['g_thm_j_dt'];
    if flag_compute_div_K:
      extract_keys += ['div_K_j_list'];

  save_data = dict(filter(lambda item: item[0] in extract_keys, output_data.items()));
  fid = open(filename,'wb'); pickle.dump(save_data,fid); fid.close();

def export_tensors(output_data):
  base_dir,base_dir_timestep,time_index,flag_verbose = tuple(map(output_data.get,
    ['base_dir','base_dir_timestep','time_index','flag_verbose']));
  flag_stochastic, = tuple(map(output_data.get,['flag_stochastic']));
  filename = '%s/tensors_%.8d.pickle'%(base_dir_timestep,time_index);
  if flag_verbose >= 2:
    print("filename = " + filename);
  save_data = {};
  extract_keys = ['bar_L', 'bar_L_indices', 'bar_K_j', 'bar_K_j_indices', 
                  'D_E', 'D_S_j', 'flag_stochastic'];
  save_data.update(dict(filter(lambda item: item[0] in extract_keys, output_data.items())));
  if flag_stochastic:
    extract_keys += ['B_j','B_j_indices'];
    if flag_compute_div_K:
      extract_keys += ['div_K_j_list'];
    save_data.update(dict(filter(lambda item: item[0] in extract_keys, output_data.items())));
  fid = open(filename,'wb'); pickle.dump(save_data,fid); fid.close();

def export_energy_flux(output_data):
  base_dir,base_dir_timestep,time_index,flag_verbose = tuple(map(output_data.get,
    ['base_dir','base_dir_timestep','time_index','flag_verbose']));
  filename = '%s/energy_flux_%.8d.pickle'%(base_dir_timestep,time_index);
  if flag_verbose >= 2:
    print("filename = " + filename);
  extract_keys = ['energy_flux_list'];

  save_data = dict(filter(lambda item: item[0] in extract_keys, output_data.items()));
  fid = open(filename,'wb'); pickle.dump(save_data,fid); fid.close();

def export_data(output_data):
 
  time_index,flag_save_system_state,skip_save_system_state, \
  flag_save_tensors,skip_save_tensors, \
  flag_save_vtk,skip_save_vtk = \
    tuple(map(output_data.get,
    ['time_index','flag_save_system_state','skip_save_system_state',
     'flag_save_tensors','skip_save_tensors','flag_save_vtk',
     'skip_save_vtk']));

  # VTK data export (of particles)
  if flag_save_vtk and time_index % skip_save_vtk == 0:
    # particle data 
    export_vtk_particle_data(output_data);

    # conc velocity and temperature fields 
    export_vtk_conc_data(output_data);

  # state
  if flag_save_system_state and time_index % skip_save_system_state == 0:
    export_system_state(output_data);

  # tensors 
  if flag_save_tensors and time_index % skip_save_tensors == 0:
    export_tensors(output_data);

  # energy flux 
  if flag_save_energy_flux and time_index % skip_save_energy_flux == 0:
    export_energy_flux(output_data);

def save_data(params,extras):
  time_index, = tuple(map(params.get,['time_index']));

  Y_n,Y_np1 = tuple(map(extras.get,['Y_n','Y_np1']));

  flag_save_vtk,skip_save_vtk,flag_save_system_state,skip_save_system_state =\
    tuple(map(params.get,
    ['flag_save_vtk','skip_save_vtk','flag_save_system_state',
     'skip_save_system_state']));

  flag_save_tensors,skip_save_tensors,flag_save_energy_flux,\
  skip_save_energy_flux = tuple(map(params.get,
    ['flag_save_tensors','skip_save_tensors','flag_save_energy_flux',
     'skip_save_energy_flux']));

  # export data for the new time step
  output_data.update({'time_index':time_index,'Y_n':Y_n,'Y_np1':Y_np1});

  if flag_save_system_state and time_index % skip_save_system_state == 0:
    # get data from update calculation 
    D_E,D_S_j = tuple(map(extras_update_state['save_data'].get,['D_E','D_S_j']));
    extras_E = {'flag_save':True};
    E = compute_E(Y_n,params,extras_E); E_kinetic_j = extras_E['E_kinetic_j'];
    U_j = extras_E['U_j']; extras_E = None; 
    extras_S_j = {'flag_save':True};
    S = compute_S(Y_n,params,extras_S_j); 
    S_j = extras_S_j['S_j']; extras_S_j = None; 
    output_data.update({'E':E,'E_kinetic_j':E_kinetic_j,'U_j':U_j,
                        'S':S,'S_j':S_j,'D_E':D_E,'D_S_j':D_S_j});
    if flag_compute_K:
      if flag_stochastic:
        g_thm_j_dt, = tuple(map(extras_update_state['save_data'].get,['g_thm_j_dt']));
        output_data.update({'g_thm_j_dt':g_thm_j_dt});
        if flag_compute_div_K:
          output_data.update({'div_K_j_list':extras_g_thm['div_K_j_list']});
      
  if flag_save_tensors and time_index % skip_save_tensors == 0:
    bar_L,bar_L_indices,D_E,D_S_j = \
      tuple(map(extras_update_state['save_data'].get,
          ['bar_L','bar_L_indices','D_E','D_S_j']));
    output_data.update({'bar_L':bar_L,'bar_L_indices':bar_L_indices,
                        'D_E':D_E,'D_S_j':D_S_j,
                        'flag_verbose':flag_verbose,
                        'flag_compute_K':flag_compute_K,
                        'flag_compute_div_K':flag_compute_div_K,
                        'flag_stochastic':flag_stochastic});
    if flag_compute_K:
      bar_K_j,bar_K_j_indices = \
        tuple(map(extras_update_state['save_data'].get,
          ['bar_K_j','bar_K_j_indices']));
      output_data.update({'bar_K_j':bar_K_j,'bar_K_j_indices':bar_K_j_indices});

      if flag_stochastic: 
        g_thm_j_dt, = \
          tuple(map(extras_update_state['save_data'].get,
            ['g_thm_j_dt']));
        output_data.update({'g_thm_j_dt':g_thm_j_dt,
                            'B_j':extras_g_thm['B_j'],
                            'B_j_indices':extras_g_thm['B_j_indices']});

        if flag_compute_div_K:
          div_K_j_list, = tuple(map(extras_update_state['save_data'].get,
                            ['div_K_j_list']));
          output_data.update({'div_K_j_list':extras_g_thm['div_K_j_list']});

  if flag_save_energy_flux and time_index % skip_save_energy_flux == 0:
    output_data.update({'energy_flux_list':extras_bar_K_j['energy_flux_list']});

  export_data(output_data);

def get_params(params_filename,params_ext=None):

  if params_ext is None:
    # get the params = {} dictionary  
    params_ext = params_filename.split(".")[-1];

  if params_ext == 'py': # run to generate params to be setup by a .py file
    loc = {'main_globals':globals()}; # collects local data from execution to return results
    exec(open(params_filename).read(),globals(),loc); 
    params = loc['params'];
  elif params_ext == 'pickle': # load using .pickle
    fid = open(params_filename,'rb'); 
    params = pickle.load(fid);   
    fid.close();
  else:
    raise Exception("Not recognized, params_ext = " + params_ext); 

  return params; 

def process_func_key(params,f_key,loc=None,flag_extras_params=False):
  # resolve the function string to
  # reference of a callable function
  # (also include reference to params in the extras)

  if loc is None:
    loc = locals(); 

  func_key = re.sub('_str$','',f_key); # remove _str at end

  if f_key in params: # set the function reference 
    f_str = params[f_key];
    if f_str in loc:
      cmd_str = "ff = loc['%s']"%f_str;
      exec(cmd_str,globals(),loc);
    else: 
      cmd_str = "ff = %s"%f_str;
      exec(cmd_str,globals(),loc);
    cmd_str = "params['%s'] = ff"%(func_key);
    exec(cmd_str); # execute the command (to set function) 

    if flag_extras_params: 
      extras_key = 'extras_' + func_key; 
      params[extras_key].update({'params':params});        
  else: # if key not in params set the function to None
    cmd_str = "params['%s'] = None"%(func_key);
    exec(cmd_str,globals(),loc); # execute the command (to set function) 

def update_state__Euler_Heun(Y_n,params,extras):

  # update using Heun's Method
  flag_compute_K,flag_stochastic,flag_test_B_j = \
    tuple(map(params.get,['flag_compute_K','flag_stochastic','flag_test_B_j']));

  deltaT,num_heat_bodies = \
    tuple(map(params.get,['deltaT','num_heat_bodies']));

  extras_bar_L,extras_bar_K_j,extras_g_thm,flag_save_data = \
    tuple(map(extras.get,['extras_bar_L','extras_bar_K_j',
                          'extras_g_thm','flag_save_data']));

  if flag_save_data is None:
    flag_save_data = False; 

  if flag_save_data:
    save_data, = tuple(map(extras.get,['save_data']));
    if save_data is None:
      save_data = {}; # create new dict (otherwise re-use)
      extras['save_data'] = save_data; 

  Y_np1 = 0.0*Y_n + Y_n; 
  hat_Y_np1 = Y_np1; 

  D_E_n = compute_D_E(Y_n,params); D_S_j_n = compute_D_S_j(Y_n,params); 

  extras_bar_L.update({'D_S_j':D_S_j_n});
  bar_L,bar_L_indices = sne.compute_bar_L_conc(Y_n,params,extras_bar_L); 
  half_bar_L__D_E_n__dt = 0.5*np.dot(bar_L,D_E_n)*deltaT; 
  sne.add_in_components(hat_Y_np1,half_bar_L__D_E_n__dt,
                        bar_L_indices['I_out'],
                        bar_L_indices['I_local_out']);

  # bar_K (t_n)
  if flag_compute_K:
    extras_bar_K_j.update({'D_E':D_E_n,'D_S_j':D_S_j_n});
    bar_K_j,bar_K_j_indices = \
      sne.compute_bar_K_j_conc(Y_n,params,extras_bar_K_j); 
    
    ii = Y_I; num_K_j = len(bar_K_j);
    for j in range(0,num_K_j): # @ optimize
      half_bar_K_j__D_S_j_n__dt = 0.5*np.dot(bar_K_j[j],D_S_j_n[j])*deltaT;
      sne.add_in_components(hat_Y_np1,half_bar_K_j__D_S_j_n__dt,
                            bar_K_j_indices['I_out'][j],
                            bar_K_j_indices['I_local_out'][j]); 

    if flag_stochastic: 
      extras_g_thm.update({'flag_save_B_j_tensors':True,
                           'flag_save_div_K_j':True,
                           'bar_K_j':bar_K_j,
                           'flag_save_dW':True,
                           'flag_use_saved_dW':False});

      g_thm_j_n_dt,B_j_indices = \
        sne.compute_g_thm_j_dt__conc(Y_n,params,extras_g_thm);
      I_local_out = B_j_indices['I_local_out']; I_out = B_j_indices['I_out'];
      for j in range(0,num_heat_bodies): # @ optimize
        half_g_thm_j_n_dt = 0.5*g_thm_j_n_dt[j];
        sne.add_in_components(hat_Y_np1,half_g_thm_j_n_dt,
                              B_j_indices['I_out'][j],
                              B_j_indices['I_local_out'][j]);

  # save data at start of the step 
  if flag_save_data:
    ss = save_data; # already linked with extras
    ss.update({'D_E':D_E_n,'D_S_j':D_S_j_n,
               'bar_L':bar_L,'bar_L_indices':bar_L_indices, 
               'bar_K_j':bar_K_j,'bar_K_j_indices':bar_K_j_indices});
    if flag_stochastic:
      ss.update({'g_thm_j_dt':g_thm_j_n_dt,'B_j_indices':B_j_indices});

  tilde_Y_np1 = 2.0*hat_Y_np1 - Y_n;

  D_E_np1 = compute_D_E(tilde_Y_np1,params); 
  D_S_j_np1 = compute_D_S_j(tilde_Y_np1,params); 

  # bar_L (t_np1) 
  extras_bar_L.update({'D_S_j':D_S_j_np1});
  bar_L,bar_L_indices = sne.compute_bar_L_conc(tilde_Y_np1,params,extras_bar_L); 
  half_bar_L__D_E_np1__dt = 0.5*np.dot(bar_L,D_E_np1)*deltaT; 
  sne.add_in_components(Y_np1,half_bar_L__D_E_np1__dt,
                        bar_L_indices['I_out'],
                        bar_L_indices['I_local_out']);
  # bar_K (t_np1)
  if flag_compute_K:
    extras_bar_K_j.update({'D_E':D_E_np1,'D_S_j':D_S_j_np1});
    bar_K_j,bar_K_j_indices = \
      sne.compute_bar_K_j_conc(tilde_Y_np1,params,extras_bar_K_j); #@@@
    
    ii = Y_I; num_K_j = len(bar_K_j);
    for j in range(0,num_K_j): # @ optimize
      half_bar_K_j__D_S_j_np1__dt = 0.5*np.dot(bar_K_j[j],D_S_j_np1[j])*deltaT;
      sne.add_in_components(Y_np1,half_bar_K_j__D_S_j_np1__dt,
                            bar_K_j_indices['I_out'][j],
                            bar_K_j_indices['I_local_out'][j]); 

    if flag_stochastic: 
      extras_g_thm.update({'flag_save_B_j_tensors':True,
                           'flag_save_div_K_j':True,
                           'bar_K_j':bar_K_j,
                           'flag_save_dW':False,
                           'flag_use_saved_dW':True});

      g_thm_j_np1_dt,B_j_indices = \
        sne.compute_g_thm_j_dt__conc(tilde_Y_np1,params,extras_g_thm); 
      I_local_out = B_j_indices['I_local_out']; I_out = B_j_indices['I_out'];
      for j in range(0,num_heat_bodies): # @ optimize
        half_g_thm_j_np1_dt = 0.5*g_thm_j_np1_dt[j]; # @@@ double-check 
        sne.add_in_components(Y_np1,half_g_thm_j_np1_dt,
                              B_j_indices['I_out'][j],
                              B_j_indices['I_local_out'][j]);

  if flag_particle_periodic:  # resets particle mod L to the domain 
    sne.map_particle_periodic(Y_np1,params);  

  return Y_np1; 

def trigger_pre__null(Y,params,extras=None):
  pass;

def trigger_post__null(Y,params,extras=None):
  pass;

# -- main 

# parse file if given on command line 
parser = argparse.ArgumentParser();
parser.add_argument('-p','--param_filename', help='parameters for the run', 
                    default=None);
parser.add_argument('-r','--restart_filename', help='data for restarting simulation', 
                    default=None);
parser.add_argument('-v','--verbosity', help='level of verbosity',
                    default=None);

args = parser.parse_args();

if args.param_filename is not None:
  print('Loading parameters from \n param_filename = %s'%args.param_filename);
  print('');

  params_ext = args.param_filename.split(".")[-1]; 
  parse_params = get_params(args.param_filename,params_ext);

  flag_params_parsed = True;
else:
  flag_params_parsed = False;
  parse_params = None; 

if args.restart_filename is not None:
  print("restart_filename = " + str(args.restart_filename));
  f = open(args.restart_filename,'rb'); restart_data = pickle.load(f); f.close();
  print("restart_data.keys() = " + str(restart_data.keys()));
  print('');
  flag_restart_sim = True;
else:
  flag_restart_sim = False;

params = {};
if flag_params_parsed:
  base_dir,gpu_device_str = tuple(
      map(parse_params.get,['base_dir','gpu_device_str']));

  # process the parameters parsed
  key_list = list(parse_params.keys());

  key_vals = tuple(map(parse_params.get,key_list));
  
  params.update(dict(zip(key_list,key_vals)));

  # process with extras
  func_key_list = ['func_compute_D_E_str','func_phi_str',
                   'func_U_q_str','func_compute_D_E_str'];
  for f_key in func_key_list:
    process_func_key(params,f_key,loc=locals(),flag_extras_params=True);

  # process without extras
  func_key_list = ['func_update_state_str','func_trigger_pre_str','func_trigger_post_str'];
  for f_key in func_key_list:
    process_func_key(params,f_key,loc=locals(),flag_extras_params=False);

else:
  raise Exception("expecting parameters specified"); 

base_name = params['base_name'];
output_dir = base_dir;

print("output_dir = " + output_dir);
sne.create_dir(output_dir);

debug_dir = output_dir + '/debug';
sne.create_dir(debug_dir);

base_dir_timestep = base_dir + '/timestep';
print("base_dir_timestep = " + base_dir_timestep);
sne.create_dir(base_dir_timestep);

params.update({'base_name':base_name,
               'base_dir':base_dir,
               'base_dir_timestep':base_dir_timestep,
               'output_dir':output_dir,
               'debug_dir':debug_dir});


np.seterr(invalid='raise');

num_dim = params['num_dim'];
flag_verbose = params['flag_verbose'];

# indices 
Y_I = create_Y_I(params);
params.update({'Y_I':Y_I});

# parameters for time-step integration
deltaT = params['deltaT']; num_timesteps = params['num_timesteps'];
t_final = deltaT*num_timesteps; num_heat_bodies = len(params['c_v']);
params.update({'num_heat_bodies':num_heat_bodies});
 
flag_save_vtk,skip_save_vtk,flag_save_system_state,skip_save_system_state =\
  tuple(map(params.get,
  ['flag_save_vtk','skip_save_vtk','flag_save_system_state',
   'skip_save_system_state']));

flag_save_tensors,skip_save_tensors,flag_save_energy_flux,\
skip_save_energy_flux = tuple(map(params.get,
  ['flag_save_tensors','skip_save_tensors','flag_save_energy_flux',
   'skip_save_energy_flux']));

# -- save parameters
filename = '%s/params.pickle'%(base_dir);
print("filename = " + filename);
fid = open(filename,'wb'); pickle.dump(params,fid); fid.close();

# function and other local data (not to be serialized)
params['func_get_parts'] = get_parts;

# -- make an archive copy of the codes in output directory 
flag_copy_codes = True;
if flag_copy_codes:
  cur_dir = os.getcwd();
  src = cur_dir + '/' + script_base_name + '.' + script_ext;
  dst = base_dir + '/' + 'archive__' + script_base_name + '.' + script_ext;
  shutil.copyfile(src, dst);
  print("Copying codes to archive:\n" + "src = " + str(src) + "\n" + "dst = " + str(dst));
  if args.param_filename is not None: # copy input parameter file (.pickle or .py)
    src = args.param_filename;
    dst = base_dir + '/' + 'archive__' + 'params' + '.' + params_ext;
    shutil.copyfile(src, dst);
    print("Copying params to archive:\n" + "src = " + str(src) + "\n" 
        + "dst = " + str(dst));

# -- initialize the state 
if flag_restart_sim:
  time_index = restart_data['time_index'];
  Y_0 = restart_data['Y_0'];
  key_list = ['time_index','Y_0'];
  key_vals = tuple(map(restart_data.get(key_list)));
  params.update(dict(zip(key_list,key_vals)));
else:
  time_index = 0;
  func_init_str,extras_init = \
    tuple(map(params.get,['func_init_str','extras_init']));
  if func_init_str is not None:
    cmd_str = "params['func_init'] = %s"%func_init_str;
    exec(cmd_str);
    func_init, = \
      tuple(map(params.get,['func_init']));
  else:
    raise Exception("need to specify the func_init"); 

  Y_0 = func_init(params,extras_init);
  params.update({'Y_0':Y_0,'time_index':0});

flag_compute_K,flag_stochastic,flag_compute_div_K,flag_particle_periodic = \
  tuple(map(params.get,['flag_compute_K','flag_stochastic',
                        'flag_compute_div_K','flag_particle_periodic']));

update_state,extras_update_state = \
  tuple(map(params.get,['func_update_state','extras_update_state']));

if extras_update_state is None:
  extras_update_state = {};

trigger_pre,extras_trigger_pre = \
  tuple(map(params.get,['func_trigger_pre','extras_trigger_pre']));

if extras_trigger_pre is None:
  extras_trigger_pre = {};

if trigger_pre is None:
  trigger_pre = trigger_pre__null; 

trigger_post,extras_trigger_post = \
  tuple(map(params.get,['func_trigger_post','extras_trigger_post']));

if trigger_post is None:
  trigger_post = trigger_post__null; 

if extras_trigger_post is None:
  extras_trigger_post = {};

flag_test_B_j, = tuple(map(params.get,['flag_test_B_j']));
if flag_test_B_j is None: flag_test_B_j = False; 

Y_0 = params['Y_0'];
disp_skip = int(num_timesteps/100); Y_n = Y_0;
output_data = {'base_dir':base_dir,
               'base_dir_timestep':base_dir_timestep,
               'time_index':time_index,
               'flag_compute_K':flag_compute_K,
               'flag_stochastic':flag_stochastic,
               'flag_compute_div_K':flag_compute_div_K,
               'skip_save_system_state':skip_save_system_state,
               'Y_n':Y_0,'Y_np1':Y_0,'Y_I':Y_I,
               'params':params, # allows for further references if needed
               'flag_verbose':flag_verbose,
               'flag_save_system_state':flag_save_system_state,
               'skip_save_system_state':skip_save_system_state,
               'flag_save_tensors':flag_save_tensors,
               'skip_save_tensors':skip_save_tensors,
               'flag_save_vtk':flag_save_vtk,
               'skip_save_vtk':skip_save_vtk};

params.update({'time_index':None}); # place holder 
extras_matrix_tensor_div = {'flag_save':True};
extras_K_visco_grad = {'flag_save':True};
extras_K_visco_dot_F = {'flag_save':True};
extras_K_heat = {'flag_save':True};
extras_bar_K_j = {'extras_matrix_tensor_div':extras_matrix_tensor_div,
                  'extras_K_visco_grad':extras_K_visco_grad,  
                  'extras_K_visco_dot_F':extras_K_visco_dot_F,
                  'extras_K_heat':extras_K_heat, 
                  'flag_save':True,
                  'flag_save_energy_flux':flag_save_energy_flux,
                  'flag_flux_check':params['flag_flux_check']};
extras_bar_L = {'D_S_j':None, 
                'flag_save':True};
extras_g_thm = {'extras_matrix_tensor_div':extras_matrix_tensor_div,
                'extras_K_visco_grad':extras_K_visco_grad,  
                'extras_K_visco_dot_F':extras_K_visco_dot_F,
                'extras_K_heat':extras_K_heat, 
                'flag_save':True};

print("params = " + str(params));

extras_update_state.update({'time_index':0,                      
                            'extras_bar_K_j':extras_bar_K_j,
                            'extras_bar_L':extras_bar_L,
                            'extras_g_thm':extras_g_thm,
                            'flag_save_data':True});

# start simulations 
Y_np1 = np.zeros(Y_n.shape); Y_np1[:] = Y_n[:]; 
for II in range(0,num_timesteps):
  t = II*deltaT; time_index = II; 
  if flag_verbose >= 1 and II % disp_skip == 0:
    print("time_index: %.6d"%time_index + "/%.6d"%num_timesteps);
  params['time_index'] = time_index; 

  # start timer
  time_start = time.time();

  # call pre-update trigger
  trigger_pre(Y_n,params,extras_trigger_pre);

  # update system state
  Y_np1 = update_state(Y_n,params,extras_update_state);

  # call post-update trigger
  trigger_post(Y_np1,params,extras_trigger_post);

  # save data 
  extras_save_data = {'Y_n':Y_n,'Y_np1':Y_np1};
  save_data(params,extras_save_data);

  # stop timer
  time_end = time.time();

  if flag_verbose >=1 and II % disp_skip == 0:
    print('elapsed time: %.1e'%(time_end - time_start) + " secs");
    print('abs(Y_n).max() = %.1e'%(np.abs(Y_n).max()));
    sys.stdout.flush();
  
  Y_n[:] = Y_np1[:]; # setup for next time-step (copy values)
  
print("-"*80);
print("done.");
print("="*80);

