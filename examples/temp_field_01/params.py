# parameter generation   
params = {};

script_basename = 'conc_field_01';

base_name = 'receptors'; run_index = 32; 
run_name = '%s_%.4d'%(base_name,run_index);
base_dir = './output/%s/%s'%(script_basename,run_name);

params.update({'base_name':base_name,
               'base_dir':base_dir});

params.update({'func_init_str':None, # setup below 
               'extras_init':{}});

params.update({
'num_dim':2,                    # number of dimensions 
'm':1.1,                        # mass of particle
'tilde_m':1.1,                  # mass of interface region
'rho':0.9,                      # mass density of fluid
'kappa_0':8.2e6,                # heat conduction (fluid) 
#'kappa_0':0.0,                 # heat conduction (fluid) 
'kappa_P_I':1.3e2,              # heat conduction (particle-interface)
#'kappa_P_I':0,                 # heat conduction (particle-interface)
'kappa_C_I':0.0,             # heat conduction per unit "volume" (conc-interface)
#'kappa_C_I':0,                 # heat conduction per unit area (conc-interface)
'kappa_C_C':8.2e4,              # conc heat conduction 
'bar_kappa':1.2e-3,             # conc conduction (concentration) 
'c0_conc':2.1e0,                # total concentration (concentration) 
'func_compute_D_E_str':"compute_D_E", 
'extras_func_compute_D_E':{},
'num_particles':1,              # number of particles  
'num_mesh_x':20,                # number grid points in x-direction
'num_mesh_y':20,                # number grid points in y-direction
#'num_mesh_x':5,                # number grid points in x-direction
#'num_mesh_y':5,                # number grid points in y-direction
'deltaX':0.1,                   # mesh spacing 
'k_B':1e-3,                     # Boltzmann's constant
'deltaT':1e-5,                  # time-step
'num_timesteps':int(16/1e-3),   # duration of the simulation
#'k_B':1e-4,                    # Boltzmann's constant
#'k_B':1e-3,                    # Boltzmann's constant
'mu':0.08,                      # fluid viscosity 
#'mu':0.0,                      # fluid viscosity 
'gamma':5,                      # viscous drag from dashpot
'gamma_particle':1e-1,           # particle drag 
#'gamma':0.0,                   # viscous drag from dashpot
'c_v':np.array([1e0,4e1,1.4]),# specific heat of each type of heat body 
                                # (fluid c_v is specific heat per unit "volume") 
'c_v_I':{'particle':0,'conc':1,'interface':2}, # indices
'flag_compute_K':True,          # dissipation contributions (or purely L)
#'flag_compute_K':False,        # dissipation contributions (or purely L)
#'flag_stochastic':False,       # fluctuation contributions
'flag_stochastic':True,         # fluctuation contributions
'flag_incompressible':False,    # incompressibility constraints
#'flag_incompressible':True,    # incompressibility constraints
'flag_ambient_drag':False,      # drag on membrane from ambient fluid
'flag_compute_div_K':False,      # compute divergence
#'flag_compute_div_K':True,      # compute divergence
'flag_verbose':2,               # how much info. to print
'flag_flux_check':False,
'flag_particle_periodic':True   # resets particles to remain in periodic domain 
});

Lx = params['num_mesh_x']*params['deltaX'];
Ly = params['num_mesh_y']*params['deltaX'];
params.update({
'func_phi_str':"phi_energy__zero", # energy acting on local particle concentration
'extras_func_phi':{'params':None}   # code updates this later
});

k_B = params['k_B']; theta_0 = 200;
func_Psi_str = "func_Psi__well_01";
extras_Psi = {'sigma_sq':0.1*0.1,
              'c2':k_B*theta_0,'X_0':np.array([1.5,1.5])};
params.update({
'func_U_q_str':"U_q_energy__conc_grad_01",  # energy acting on local particle
'extras_func_U_q':{'params':None,        # code updates this later
                   'func_Psi_str':func_Psi_str,
                   'extras_Psi':extras_Psi}, # extras for energy
});

num_timesteps = params['num_timesteps'];

params.update({
'func_update_state_str':"update_state__Euler_Heun", # numerical integrator 
#'func_update_state_str':"update_state__Euler_Marayuma", # numerical integrator 
'extras_update_state':{'params':None}
});


params.update({
'flag_save_vtk':True,
'skip_save_vtk':int(num_timesteps/1000),
#'skip_save_vtk':int(num_timesteps/1000),
'flag_save_system_state':True,
'skip_save_system_state':int(num_timesteps/1000),
'flag_save_tensors':False,
'skip_save_tensors':int(num_timesteps/1000),
'flag_save_energy_flux':True,
'skip_save_energy_flux':int(num_timesteps/1000),
});

# =============================
def init_model__receptors(params,extras=None):
  Y_I = params['Y_I'];
  #locals().update(**Y_I); # adds indices to local variables 
  num_mesh_x,num_mesh_y,deltaX = \
    tuple(map(params.get,['num_mesh_x','num_mesh_y','deltaX']));
  Lx = num_mesh_x*deltaX; Ly = num_mesh_y*deltaX;

  I1_particle_q,I2_particle_q = tuple(map(
    Y_I.get,['I1_particle_q','I2_particle_q']));
  I1_particle_theta,I2_particle_theta = tuple(map(
    Y_I.get,['I1_particle_theta','I2_particle_theta']));

  I1_conc_q,I2_conc_q = tuple(map(
    Y_I.get,['I1_conc_q','I2_conc_q']));
  I1_conc_theta,I2_conc_theta = tuple(map(
    Y_I.get,['I1_conc_theta','I2_conc_theta']));

  I1_interface_q,I2_interface_q = tuple(map(    
    Y_I.get,['I1_interface_q','I2_interface_q']));
  I1_interface_theta,I2_interface_theta = tuple(map(
    Y_I.get,['I1_interface_theta','I2_interface_theta']));

  num_particle_q = I2_particle_q - I1_particle_q;
  num_particle_theta = I2_particle_theta - I1_particle_theta;

  num_conc_q = I2_conc_q - I1_conc_q;
  num_conc_theta = I2_conc_theta - I1_conc_theta;

  num_interface_q = I2_interface_q - I1_interface_q;
  num_interface_theta = I2_interface_theta - I1_interface_theta;

  Y_0 = np.zeros(Y_I['I2_interface_theta']); ii = Y_I
  # particle
  num_particles = params['num_particles']; 
  particle_q = np.zeros(num_particles*num_dim);
  #particle_q[0*num_dim:1*num_dim] = np.array([Lx/2.0,Ly/2.0]);
  particle_q[0*num_dim:1*num_dim] = np.array([1.0,1.0]);
  particle_theta = np.zeros(num_particles) + 3.0;

  sne.set_comp(Y_0,'I1_particle_q','I2_particle_q',Y_I,particle_q);
  sne.set_comp(Y_0,'I1_particle_theta','I2_particle_theta',Y_I,particle_theta);

  # conc
  conc_q = np.zeros(num_conc_q) + 1.1;
  x1,x2,Lx,Ly = sne.get_mesh_coord(params); k1 = 1.0; k2 = 1.0;
  xx = np.vstack((x1,x2)).T;
  #conc_q[:] = 2.0 + 0.5*np.sin(2.0*np.pi*k1*x1/Lx);
 
  X_0 = np.array([1.5,1.5]); XX_0 = np.expand_dims(X_0,0);
  #sigma_sq = 0.2*0.2;
  conc_q[:] = 1.0; # + np.exp(-np.sum(np.power((xx - XX_0),2),1)/(2.0*sigma_sq)); 

  #i0 = num_mesh_x//2; j0 = num_mesh_y//2; 
  #I0 = j0*num_mesh_x*num_dim + i0*num_dim; d = 0;
  #sigma_sq = 0.2*0.2;
  conc_theta = np.zeros(num_conc_theta);
  #conc_theta = 100*(2.0 + np.exp(-np.sum(np.power((xx - XX_0),2),1)/(2.0*sigma_sq))); 
  vec_k = np.array([[0.5/Lx,0.0/Ly]]);
  #vec_k = vec_k/np.sqrt(np.sum(np.power(vec_k,2))); # unit vector
  conc_theta[:] = 3.0*(1.0 + 0.6*np.sin(2.0*np.pi*np.sum(vec_k*xx,1))); 
  #conc_theta = 0*conc_theta + 1.0; 
  #conc_theta[:] = 1.0 + 0.1*np.sin(2.0*np.pi*k1*x1/Lx)*np.sin(2.0*np.pi*k2*x2/Ly);
  #conc_theta[:] = 3.0 + 0.0*x1; #3.0 + 2.5*np.sin(2.0*np.pi*k1*x1/Lx);

  #k1 = 1; k2 = 1;
  #conc_theta += 0.5*np.sin(2*np.pi*k1*x1/Lx)*np.sin(2*np.pi*k2*x2/Ly);
  #i0 = num_mesh_x//2; j0 = num_mesh_y//2; 
  #I0 = j0*num_mesh_x + i0; d = 0;
  #conc_theta[I0] += 1.0*(1.0/(deltaX*deltaX));
  #conc_theta[I0] += 2.0;

  sne.set_comp(Y_0,'I1_conc_q','I2_conc_q',Y_I,conc_q);
  sne.set_comp(Y_0,'I1_conc_theta','I2_conc_theta',Y_I,conc_theta);

  # interface
  interface_q = np.zeros(num_interface_q) + 1.2;
  interface_theta = np.zeros(num_interface_theta) + particle_theta;
  #interface_theta = np.zeros(num_interface_theta) + 1.0;

  sne.set_comp(Y_0,'I1_interface_q','I2_interface_q',Y_I,interface_q);
  sne.set_comp(Y_0,'I1_interface_theta','I2_interface_theta',Y_I,interface_theta);

  return Y_0; 


# add to the variables in main code
main_globals.update({'init_model__receptors':init_model__receptors});

params.update({'func_init_str':'init_model__receptors',
               'extras_init':{}});


