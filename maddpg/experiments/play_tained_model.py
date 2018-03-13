#!/usr/bin/env python
import os
import sys
import re


def get_scenario_name(model):
    repat = re.compile('^exp_([!-~]+)_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$')
    for dir_name in model.split('/'):
        m = repat.match(dir_name)
        if m:
            scenario_name = m.group(1)
            return scenario_name


trained_model = sys.argv[1]
scenario = get_scenario_name(trained_model)

os.system('python train.py --display --scenario %s --load-model %s'
          % (scenario, trained_model))
