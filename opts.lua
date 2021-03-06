#!/usr/bin/env th
--[[
Parse the arguments.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
  modify  -  Feng Zhou (zhfe99@gmail.com), 2015-11
]]--

local lib = require('lua_lib')

local opts = {}

----------------------------------------------------------------------
-- Parse options.
--
-- Input
--   arg   -  input
--   mode  -  mode, 'train' | 'test' | 'demo'
--
-- Output
--   opt   -  option
--   con   -  configuration
function opts.parse(arg, mode)

  -- data
  local dbe = arg[1]
  local ver = arg[2]
  local conNm = arg[3]
  table.remove(arg, 1)
  table.remove(arg, 1)
  table.remove(arg, 1)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 Training / Testing script')
  cmd:text()
  cmd:text('Options:')
  cmd:addTime()
  cmd:option('-seed', 2, 'manually set RNG seed')
  cmd:option('-prL', 4, 'prompt level')
  cmd:option('-threads', 8, '#threads')
  cmd:option('-nGpu', 1, '#GPUs')
  cmd:option('-shuffle', true, 'shuffle training samples')
  cmd:option('-cmp', true, 'using compressed data or not')
  cmd:option('-local', true, 'using local data or not')
  cmd:option('-debG', 0, 'debug grid')
  cmd:option('-debI', 0, 'debug image')
  cmd:option('-epo', 1, 'epoch id for testing')
  opt = cmd:parse(arg or {})

  -- data
  local th_lst = require('lua_th.th_lst')
  opt.PATH = th_lst.dbeInfoPath(dbe, ver)
  opt.DATA = th_lst.dbeInfoData(opt.PATH)
  opt.CONF = th_lst.dbeInfoConf(opt.PATH, conNm)

  -- configuration
  local con = dofile(opt.CONF.protTr)

  -- log
  cmd:log(opt.CONF.logPath .. '_' .. mode)
  lib.prSet(opt.prL)
  lib.prIn('opts.parse', '%s %s %s', dbe, ver, conNm)

  -- gpu
  if con.nGpu then
    opt.nGpu = con.nGpu
  end
  cutorch.setDevice(1)

  -- other
  torch.setnumthreads(opt.threads)
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.manualSeed(opt.seed)

  lib.prTab(opt, 'opt')
  lib.prTab(con, 'con')

  lib.prOut()
  return opt, con
end

return opts
