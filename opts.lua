#!/usr/bin/env th
-- Parse the arguments.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

local lib = require('lua_lib')

local M = {}

----------------------------------------------------------------------
-- Parse options.
--
-- Input
--   arg   -  input
--   mode  -  mode, 'train' | 'test' | 'demo'
--
-- Output
--   opt   -  option
--   solConf  -  solver configuration
function M.parse(arg, mode)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 Training / Testing script')
  cmd:text()
  cmd:text('Options:')
  cmd:addTime()
  cmd:option('-seed', 2, 'manually set RNG seed')
  cmd:option('-dbe', 'bird', 'database name')
  cmd:option('-ver', 'v1', 'version')
  cmd:option('-con', 'alex', 'configuration')
  cmd:option('-threads', 8, '#threads')
  cmd:option('-nGpu', 1, '#GPUs')
  cmd:option('-shuffle', true, 'shuffle training samples')
  cmd:option('-cmp', true, 'compress or not')
  cmd:option('-local', true, 'using local data')
  cmd:option('-deb', false, 'debug mode')
  opt = cmd:parse(arg or {})

  local dbe = opt.dbe
  local ver = opt.ver
  local con = opt.con

  -- data
  local th_lst = require('lua_th.th_lst')
  opt.PATH = th_lst.dbeInfoPath(dbe, ver)
  opt.DATA = th_lst.dbeInfoData(opt.PATH)
  opt.CONF = th_lst.dbeInfoConf(opt.PATH, con)

  -- configuration
  local solConf = dofile(opt.CONF.protTr)

  -- log
  cmd:log(opt.CONF.logPath .. '_' .. mode)

  -- gpu
  if solConf.nGpu then
    opt.nGpu = solConf.nGpu
  end
  cutorch.setDevice(1)

  -- other
  torch.setnumthreads(opt.threads)
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.manualSeed(opt.seed)

  lib.prTab(opt, 'opt')
  lib.prTab(solConf, 'solConf')

  return opt, solConf
end

return M
