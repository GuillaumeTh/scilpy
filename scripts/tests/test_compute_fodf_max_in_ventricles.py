#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

def test_help_option(script_runner):
    ret = script_runner.run('scil_compute_fodf_max_in_ventricles.py', '--help')
    assert ret.success