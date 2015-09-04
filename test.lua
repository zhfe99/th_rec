#!/usr/bin/env th
-- Train using Torch.
--
-- Example
--   export CUDA_VISIBLE_DEVICES=0,1,2,3
--   ./test.lua -dbe imgnet -ver v2 -con alx
--   ./test.lua -ver v1 -con alexS1 -deb
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09

require('torch')
local th = require('lua_th')
local lib = require('lua_lib')
local opts = require('opts')
local net = require('net')
local dp = require('dp_load')
local th_step = require('lua_th.th_step')

-- option
local opt, con = opts.parse(arg, 'train')

-- network
local model, loss, modelSv, modss, optStat = net.new(con, opt)
local optimator

-- data provider
local _, teDB = dp.init(opt, con)

-- load
model = torch.load(opt.CONF.modPath .. '_' .. opt.epo .. '.t7')

-- test
th_step.ford(teDB, false, opt.epo, opt, con, model, loss, modss, optStat, optimator)
