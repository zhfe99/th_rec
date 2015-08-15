#!/usr/bin/env th
-- Parse the arguments.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-15-2015

local lib = require('lua_lib')

local M = {}

function M.parse(arg, mode)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 Training / Testing script')
  cmd:text()
  cmd:text('Options:')
  cmd:addTime()
  cmd:option('-seed', 2, 'manually set RNG seed')
  cmd:option('-dbe', 'car', 'database name')
  cmd:option('-ver', 'v1c', 'version')
  cmd:option('-con', 'alex', 'configuration')
  cmd:option('-testonly', false, 'Just test loaded net on validation set')
  cmd:option('-threads', 8, 'number of threads')
  cmd:option('-type', 'cuda', 'float or cuda')
  cmd:option('-gpu', '0', 'gpu id, could be multiple')
  cmd:option('-shuffle', true, 'shuffle training samples')
  cmd:option('-cmp', true, 'compress or not')
  opt = cmd:parse(arg or {})

  local dbe = opt.dbe
  local ver = opt.ver
  local con = opt.con

  -- data
  local th_lst = require('lua_th.th_lst')
  opt.PATH = th_lst.dbeInfoPath(dbe, ver)
  opt.DATA = th_lst.dbeInfoData(opt.PATH)
  opt.CONF = th_lst.dbeInfoConf(opt.PATH, con)

  -- folder
  opt.network = string.format('./model/%s_%s_%s', dbe, ver, con)

  -- log
  cmd:log(opt.CONF.logPath .. '_' .. mode)

  -- cuda
  opt.gpus = lib.str2idx(opt.gpu)
  torch.setnumthreads(opt.threads)
  cutorch.setDevice(opt.gpus[1] + 1)
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.manualSeed(opt.seed)

  return opt
end

return M
