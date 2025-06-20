# parameter generation   
params = {};

script_basename = 'conc_field_01';

base_name = "energy_well_11_KBT_001500en7"; run_index = 5; 
run_name = '%s__%.4d'%(base_name,run_index);
base_dir = './output/%s/%s'%(script_basename,run_name);

params.update({'base_name':base_name,
               'base_dir':base_dir});

flag_debug = False; 
params.update({'flag_debug':flag_debug});

params.update({'func_init_str':None, 
               'extras_init':{}});

params.update({
'num_dim':2,                    
'm':1.1,                        
'tilde_m':1.1,                  
'rho':0.9,                      
'kappa_0':8.2e6,                
'kappa_P_I':2.3e3/4.0,          
'kappa_C_I':3.02e3,             
'kappa_C_C':2.1e-3,             
'bar_kappa':1.2e-3,             
'c0_conc':2.1e0,                
'func_compute_D_E_str':"compute_D_E", 
'extras_func_compute_D_E':{},
'num_particles':1,              
'num_mesh_x':20,                
'num_mesh_y':20,                
'deltaX':0.1,                   
'k_B':1.000e-05,                     
'deltaT':3e-3,                  
'num_timesteps':128000,   
'mu':0.08,                      
'gamma_particle':1e-1,           
'c_v':np.array([9.3e2,1.3e4,1.4e2]),
'c_v_I':{'particle':0,'conc':1,'interface':2}, 
'flag_compute_K':True,          
'flag_stochastic':True,         
'flag_compute_div_K':False,      
'flag_verbose':2,               
'flag_flux_check':False,
'flag_particle_periodic':True   
});

Lx = params['num_mesh_x']*params['deltaX'];
Ly = params['num_mesh_y']*params['deltaX'];
params.update({
'func_phi_str':"phi_energy__zero", 
'extras_func_phi':{'params':None}   
});

k_B = params['k_B']; 
theta_0=3.000e+00;
c2=1.500e-04; 
func_Psi_str = "func_Psi__well_02";
extras_Psi = {'sigma_sq':0.2**2,
              'c2':c2,'X_0':np.array([5.0/3.0,1.0])};
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

if flag_debug == False:
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
else:
  params.update({
  'flag_save_vtk':True,
  'skip_save_vtk':1,
  'flag_save_system_state':True,
  'skip_save_system_state':1,
  'flag_save_tensors':False,
  'skip_save_tensors':1,
  'flag_save_energy_flux':True,
  'skip_save_energy_flux':1
  });

def init_model__energy_well(params,extras=None):

  if extras is None:
    raise Exception("expected extras dictionary specified");
  else:
    ref_data, = tuple(map(extras.get,['ref_data']));

  Y_I = params['Y_I'];
  Y_0 = np.zeros(Y_I['I2_interface_theta']); ii = Y_I

  sne.set_comp(Y_0,'I1_particle_q','I2_particle_q',Y_I,ref_data['particle_q_0']);
  sne.set_comp(Y_0,'I1_particle_theta','I2_particle_theta',Y_I,ref_data['particle_theta_0']);

  sne.set_comp(Y_0,'I1_conc_q','I2_conc_q',Y_I,ref_data['conc_q_0']);
  sne.set_comp(Y_0,'I1_conc_theta','I2_conc_theta',Y_I,ref_data['conc_theta_0']);

  sne.set_comp(Y_0,'I1_interface_q','I2_interface_q',Y_I,ref_data['interface_q_0']);
  sne.set_comp(Y_0,'I1_interface_theta','I2_interface_theta',Y_I,ref_data['interface_theta_0']);

  return Y_0; 


def trigger_post__energy_well(Y_np1,params,extras):

  flag_debug = False; 

  Y_I,time_index,base_dir_timestep,flag_verbose = tuple(map(params.get,
      ['Y_I','time_index','base_dir_timestep','flag_verbose']));
 
  R_sq,ref_data = tuple(map(extras.get,['R_sq','ref_data']));
  
  i1 = Y_I['I1_particle_q']; i2 = Y_I['I2_particle_q'];
  X = Y_np1[i1:i2];
  X_0 = ref_data['particle_q_0'];

  r_sq = np.sum(np.power(X - X_0,2));
  if flag_debug:
    print("r_sq = " + str(r_sq));

  if r_sq >= R_sq: 
    tt = {'X':X,'time_index':time_index};

    filename = '%s/trigger_post__%.8d.pickle'%(base_dir_timestep,time_index);
    if flag_verbose >= 2:
      print("filename = " + filename);
    fid = open(filename,'wb'); pickle.dump(tt,fid); fid.close();

    sne.set_comp(Y_np1,'I1_particle_q','I2_particle_q',Y_I,ref_data['particle_q_0']);
    sne.set_comp(Y_np1,'I1_particle_theta','I2_particle_theta',Y_I,ref_data['particle_theta_0']);

    sne.set_comp(Y_np1,'I1_conc_q','I2_conc_q',Y_I,ref_data['conc_q_0']);
    sne.set_comp(Y_np1,'I1_conc_theta','I2_conc_theta',Y_I,ref_data['conc_theta_0']);

    sne.set_comp(Y_np1,'I1_interface_q','I2_interface_q',Y_I,ref_data['interface_q_0']);
    sne.set_comp(Y_np1,'I1_interface_theta','I2_interface_theta',Y_I,ref_data['interface_theta_0']);


def get_ref_data(params,extras=None):

  if extras is None:
    raise Exception("expected extras dictionary specified");

  Y_I = create_Y_I(params);  
  num_dim = params['num_dim'];
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

  num_particles = 1;
  particle_q = X_0 = np.array([5.0/3.0,1.0]); 
  particle_theta = np.zeros(num_particles) + 3.0;

  XX_0 = np.expand_dims(X_0,0); 

  conc_q = np.zeros(num_conc_q) + 1.1;
  x1,x2,Lx,Ly = sne.get_mesh_coord(params); k1 = 1.0; k2 = 1.0;
  xx = np.vstack((x1,x2)).T;
  
  conc_q[:] = 1.0; 

  conc_sigma_sq = extras['conc_sigma_sq'];
  theta_0 = extras['theta_0'];
  c3 = extras['c3'];

  conc_theta = np.zeros(num_conc_theta);
  
  sigma_sq = conc_sigma_sq; 
  conc_theta = theta_0*(1.0 + c3*np.exp(-np.sum(np.power((xx - XX_0),2),1)/(2.0*sigma_sq))); 

  interface_q = 0*particle_q;
  interface_theta = np.zeros(num_interface_theta) + 1.2;

  ref_data = {};
  ref_data['extras'] = extras;
  ref_data['particle_q_0'] = particle_q;
  ref_data['particle_theta_0'] = particle_theta;

  ref_data['conc_q_0'] = conc_q;
  ref_data['conc_theta_0'] = conc_theta;

  ref_data['interface_q_0'] = interface_q;
  ref_data['interface_theta_0'] = interface_theta;

  return ref_data; 

main_globals.update({'init_model__energy_well':init_model__energy_well,
                     'trigger_post__energy_well':trigger_post__energy_well});

extras_ref_data = {'theta_0':theta_0,'c2':c2,'c3':1.071e+01,'conc_sigma_sq':0.3**2};
ref_data = get_ref_data(params,extras_ref_data);

params.update({'func_init_str':'init_model__energy_well',
               'extras_init':{'ref_data':ref_data},
               'func_trigger_post_str':'trigger_post__energy_well',
               'extras_trigger_post':{'R_sq':0.3**2, 
               'ref_data':ref_data}});

