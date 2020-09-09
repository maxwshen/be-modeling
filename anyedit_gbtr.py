# 
from __future__ import division
import _config
import sys, os, fnmatch, datetime, subprocess, pickle
sys.path.append('/home/unix/maxwshen/')
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Default params
inp_dir = _config.OUT_PLACE + 'anyedit_data/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

import _data

nts = list('ACGT')
nt_to_idx = {nts[s]: s for s in range(len(nts))}

##
# Load train test
##
def get_traintest_package(X_all, data, lib_nm):
  rs = 0
  tt_df = pd.read_csv(_config.OUT_PLACE + 'gen_traintest_idxs/%s_%s.csv' % (lib_nm, rs), index_col = 0)
  nms_train = set(tt_df[tt_df['Category'] == 'Train']['Name'])
  nms_test = set(tt_df[tt_df['Category'] == 'Test']['Name'])

  dd = defaultdict(list)
  for idx, row in data.iterrows():
    nm = row['Name (unique)']
    if nm in nms_train:
      cat = 'train'
    elif nm in nms_test:
      cat = 'test'

    dd['x_' + cat].append(X_all[idx])
    dd['y_' + cat].append(row['Y'])
    dd['w_' + cat].append(row['Y_weight'])
    dd['nm_' + cat].append(nm)

  package = (
    dd['x_train'],
    dd['x_test'],
    dd['y_train'],
    dd['y_test'],
    dd['w_train'],
    dd['w_test'],
    dd['nm_train'],
    dd['nm_test'],
  )

  return package

'''
  Featurization
'''
ohe_encoder = {
  'A': [1, 0, 0, 0],
  'C': [0, 1, 0, 0],
  'G': [0, 0, 1, 0],
  'T': [0, 0, 0, 1],
}
def one_hot_encode(seq):
  ohe = []
  for nt in seq:
    ohe += ohe_encoder[nt]
  return ohe
  
def get_one_hot_encoder_nms(start_pos, end_pos):
  nms = []
  nts = list('ACGT')
  for pos in range(start_pos, end_pos + 1):
    for nt in nts:
      nms.append('%s%s' % (nt, pos))
  return nms

dint_encoder = {
  'AA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AC': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AG': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'AT': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CA': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CC': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CG': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  'CT': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  'GA': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  'GC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  'GG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  'GT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  'TA': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
  'TC': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
  'TG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,],
  'TT': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
}
def dinucleotide_encode(seq):
  ohe = []
  for idx in range(len(seq) - 1):
    ohe += dint_encoder[seq[idx : idx + 2]]
  return ohe

def get_dinucleotide_nms(start_pos, end_pos):
  nms = []
  dints = sorted(list(dint_encoder.keys()))
  for pos in range(start_pos, end_pos):
    for dint in dints:
      nms.append('%s%s' % (dint, pos))
  return nms


def featurize(data, exp_nm, seq_col):
  X_all = []
  start_pos, end_pos = -9, 21   # go up to N in NGG

  for idx, row in data.iterrows():
    x_input = row[seq_col]
    # zero_idx = _data.pos_to_idx(0, exp_nm)
    zero_idx = _data.pos_to_idx_safe(0, exp_nm, row['Name (unique)'])
    seq = x_input[zero_idx + start_pos : zero_idx + end_pos + 1]
    assert len(seq) == 31

    curr_x = []

    # One hot encoding
    curr_x += one_hot_encode(seq)

    # Dinucleotides
    curr_x += dinucleotide_encode(seq)

    # Sum nucleotides
    features = [
      seq.count('A'),
      seq.count('C'),
      seq.count('G'),
      seq.count('T'),
      seq.count('G') + seq.count('C'),
    ]
    curr_x += features

    # Melting temp
    from Bio.SeqUtils import MeltingTemp as mt
    features = [
      mt.Tm_NN(seq),
      mt.Tm_NN(seq[-5:]),
      mt.Tm_NN(seq[-13:-5]),
      mt.Tm_NN(seq[-21:-13]),
    ]
    curr_x += features

    # Store
    X_all.append(np.array(curr_x))

  ohe_nms = get_one_hot_encoder_nms(start_pos, end_pos)
  dint_nms = get_dinucleotide_nms(start_pos, end_pos)
  sum_nms = ['Num. A', 'Num. C', 'Num. G', 'Num. T', 'Num. GC']
  mt_nms = ['Tm full', 'Tm -5', 'Tm -13 to -5', 'Tm -21 to -13']
  param_nms = ['x_%s' % (ft_nm) for ft_nm in ohe_nms + dint_nms + sum_nms + mt_nms]

  return (np.array(X_all), param_nms)


'''
  Statistics
'''
def weighted_mean(x, w):
  return np.sum(x * w) / np.sum(w)

def weighted_cov(x, y, w):
  return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

def weighted_pearsonr(x, y, w):
  x, y, w = np.array(x), np.array(y), np.array(w)
  return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))

'''
  Primary
'''
def train_models(exp_nm, data, ml_task, seq_col):
  # Prepare models and data

  if ml_task == 'regress_nonzero':
    evals = {
      'spearmanr': lambda t, p, w: spearmanr(t, p)[0],
      'pearsonr': lambda t, p, w: pearsonr(t, p)[0],
      'pearsonr weighted': lambda t, p, w: weighted_pearsonr(t, p, w),
      'r2_score weighted': lambda t, p, w: sklearn.metrics.r2_score(t, p, sample_weight = w),
      'r2_score unweighted': lambda t, p, w: sklearn.metrics.r2_score(t, p),
    }

  data = data[~np.isnan(data['Y'])]
  data = data.reset_index(drop = True)

  # Prepare additional features
  package = featurize(data, exp_nm, seq_col)
  (X_all, param_nms) = package
  import code; code.interact(local=dict(globals(), **locals()))

  # Train test split
  lib_nm = _data.get_lib_nm(exp_nm)
  package = get_traintest_package(X_all, data, lib_nm)
  (x_train, x_test, y_train, y_test, w_train, w_test, nms_train, nms_test) = package

  # Train models
  ms_dd = defaultdict(list)
  ms_dd['Name'].append(exp_nm)

  model_nm = 'GBTR'

  # Hyperparameter optimization
  '''
    Approx 20 seconds per fit.
    5 * 3 * 6 * 5 * 20 seconds = 2.5 hours
  '''
  from sklearn.model_selection import GridSearchCV
  hyperparameters = {
    'n_estimators': [100, 250, 500],
    'min_samples_leaf': [2, 5],
    'max_depth': [2, 3, 4, 5],
  }
  # hyperparameters = {
  #   'n_estimators': [100, 200],
  #   'min_samples_leaf': [1],
  #   'max_depth': [3, 4],
  # }

  model = GridSearchCV(
    GradientBoostingRegressor(),
    hyperparameters,
    cv = 5,
    verbose = True,
  )

  model.fit(x_train, y_train, sample_weight = w_train)

  gscv_df = pd.DataFrame(model.cv_results_)
  gscv_df.to_csv(out_dir + '%s_hyperparamresults.csv' % (exp_nm))

  with open(out_dir + '%s_bestmodel.pkl' % (exp_nm), 'wb') as f:
    pickle.dump(model.best_estimator_, f)

  pred_train = model.predict(x_train)
  pred_test = model.predict(x_test)

  # Store model performance stats in modelstats_dd
  for ml_eval_nm in evals:
    eval_f = evals[ml_eval_nm]

    try:
      ev = eval_f(y_train, pred_train, w_train)
    except ValueError:
      ev = np.nan
    ms_dd['%s %s train' % (model_nm, ml_eval_nm)].append(ev)

    try:
      ev = eval_f(y_test, pred_test, w_test)
    except ValueError:
      ev = np.nan
    ms_dd['%s %s test' % (model_nm, ml_eval_nm)].append(ev)

  # Record predictions in data
  pred_df = pd.DataFrame({
    'Name (unique)': nms_train + nms_test,
    'y_pred_%s' % (model_nm): list(pred_train) + list(pred_test),
    'TrainTest_%s' % (model_nm): ['train'] * len(nms_train) + ['test'] * len(nms_test)
  })
  data = data.merge(pred_df, on = 'Name (unique)')

  ms_df = pd.DataFrame(ms_dd)
  ms_df = ms_df.reindex(sorted(ms_df.columns), axis = 1)
  return (ms_df, data)

##
# IO
##
def save_results(exp_nm, ml_task, results):
  (model_stats, pred_obs_df) = results

  model_stats.to_csv(out_dir + '%s_%s_model_stats.csv' % (exp_nm, ml_task))

  pred_obs_df.to_csv(out_dir + '%s_%s_df.csv' % (exp_nm, ml_task))

  return

##
# Primary logic
##
def gather_statistics(exp_nm):
  # Load data
  data = pd.read_csv(inp_dir + '%s.csv' % (exp_nm), index_col = 0)
  
  seq_col = [col for col in data.columns if 'Sequence context' in col][0]

  # # Gather statistics
  for ml_task in ['regress_nonzero']:
    print(ml_task)
    results = train_models(exp_nm, data, ml_task, seq_col)
    save_results(exp_nm, ml_task, results)

  return



##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  # Generate qsubs only for unfinished jobs
  treat_control_df = pd.read_csv(_config.DATA_DIR + 'treatment_control_design.csv', index_col = 0)

  data_nms = [s.replace('.csv', '') for s in os.listdir(inp_dir)]
  num_scripts = 0

  # editors = ['ABE8']
  editors = _data.cgbes

  # editors = _data.main_base_editors
  celltypes = ['HEK293T', 'mES']
  libraries = ['12kChar']
  # libraries = ['12kChar', 'CtoT', 'AtoG', 'CtoGA']

  for celltype in celltypes:
    for lib_nm in libraries:
      for editor_nm in editors:

        data_nm = f'{celltype}_{lib_nm}_{editor_nm}'

        out_fn = out_dir + f'{data_nm}_regress_nonzero_df.csv'
        if os.path.isfile(out_fn): continue

        command = 'python %s.py %s' % (NAME, data_nm)
        script_id = NAME.split('_')[0]

        # Write shell scripts
        sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, data_nm)
        with open(sh_fn, 'w') as f:
          f.write('#!/bin/bash\n%s\n' % (command))
        num_scripts += 1

        # Write qsub commands
        qsub_commands.append('qsub -V -P regevlab -l h_rt=24:00:00,h_vmem=2G -wd %s %s &' % (_config.SRC_DIR, sh_fn))

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))

  subprocess.check_output('chmod +x %s' % (commands_fn), shell = True)

  print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return

##
# Main
##
@util.time_dec
def main(exp_nm = ''):
  print(NAME)
  
  gather_statistics(exp_nm)


  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(exp_nm = sys.argv[1])
  else:
    gen_qsubs()
    # main()