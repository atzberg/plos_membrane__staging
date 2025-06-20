# parameter generation   
params = {};

script_basename = 'conc_field_01';

base_name = 'conc_grad'; run_index = 1; 
run_name = '%s_%.4d'%(base_name,run_index);
base_dir = './output/%s/%s'%(script_basename,run_name);

params.update({'base_name':base_name,
               'base_dir':base_dir});

params.update({'func_init_str':None, # setup below 
               'extras_init':{}});

params.update({
'num_dim':2,                    
'm':1.1,                        
'tilde_m':1.1,                  
'rho':0.9,                      
'kappa_0':8.2e6,                
'kappa_P_I':1.3e2,              
'kappa_C_I':1.02e2,             
'kappa_C_C':8.2e6,              
'bar_kappa':0.001,       
'c0_conc':2.1e0,                
'func_compute_D_E_str':"compute_D_E", 
'extras_func_compute_D_E':{},
'num_particles':1,              
'num_mesh_x':20,                
'num_mesh_y':20,                
'deltaX':0.1,                   
'k_B':1e-5,                     
'deltaT':1e-3,                  
'num_timesteps':16000,   
'gamma_particle':91.0,      
'c_v':np.array([1.2,1.3e2,1.4]),
'c_v_I':{'particle':0,'conc':1,'interface':2}, 
'flag_compute_K':True,          
'flag_stochastic':True,         
'flag_ambient_drag':False,      
'flag_compute_div_K':False,      
'flag_verbose':2,               
'flag_flux_check':False,
'flag_particle_periodic':True   
});

Lx = params['num_mesh_x']*params['deltaX'];
Ly = params['num_mesh_y']*params['deltaX'];
params.update({
'func_phi_str':"phi_energy__eta_gauss_01", 
'extras_func_phi':{'params':None,   
                   'k1':1.1,  
                   'sigma_sq':0.2*0.2} 
});

func_Psi_str = "func_Psi__zero";
extras_Psi = {};
params.update({
'func_U_q_str':"U_q_energy__conc_grad_01",  
'extras_func_U_q':{'params':None,        
                   'func_Psi_str':func_Psi_str,
                   'extras_Psi':extras_Psi}, 
});

num_timesteps = params['num_timesteps'];

params.update({
'func_update_state_str':"update_state__Euler_Heun", 
'extras_update_state':{'params':None}
});

params.update({
'flag_save_vtk':True,
'skip_save_vtk':int(num_timesteps/100),
'flag_save_system_state':True,
'skip_save_system_state':int(num_timesteps/100),
'flag_save_tensors':False,
'skip_save_tensors':int(num_timesteps/100),
'flag_save_energy_flux':True,
'skip_save_energy_flux':int(num_timesteps/100),
});

def init_model__conc_grad(params,extras):
  Y_I = params['Y_I'];
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
  
  num_particles = params['num_particles']; 
  particle_q = np.zeros(num_particles*num_dim);
  particle_q[0*num_dim:1*num_dim] = np.array([Lx/2.0,Ly/2.0]);
  
  particle_theta = np.zeros(num_particles) + 3.0;

  sne.set_comp(Y_0,'I1_particle_q','I2_particle_q',Y_I,particle_q);
  sne.set_comp(Y_0,'I1_particle_theta','I2_particle_theta',Y_I,particle_theta);
  
  conc_q = np.zeros(num_conc_q) + 1.1;
  x1,x2,Lx,Ly = sne.get_mesh_coord(params); k1 = 1.0; k2 = 1.0;
  xx = np.vstack((x1,x2)).T;
 
  X_0 = np.array([1.5,1.5]); XX_0 = np.expand_dims(X_0,0);
  sigma_sq = 0.2*0.2;
  conc_q[:] = 1.0 + np.exp(-np.sum(np.power((xx - XX_0),2),1)/(2.0*sigma_sq)); 
  
  conc_theta = np.zeros(num_conc_theta);
  
  conc_theta[:] = 3.0 + 0.0*x1; 

  sne.set_comp(Y_0,'I1_conc_q','I2_conc_q',Y_I,conc_q);
  sne.set_comp(Y_0,'I1_conc_theta','I2_conc_theta',Y_I,conc_theta);
  
  interface_q = np.zeros(num_interface_q) + 1.2;
  interface_theta = np.zeros(num_interface_theta) + particle_theta;

  sne.set_comp(Y_0,'I1_interface_q','I2_interface_q',Y_I,interface_q);
  sne.set_comp(Y_0,'I1_interface_theta','I2_interface_theta',Y_I,interface_theta);

  return Y_0; 

# add to the variables in main code
main_globals.update({'init_model__conc_grad':init_model__conc_grad});

params.update({'func_init_str':'init_model__conc_grad',
               'extras_init':{}});

