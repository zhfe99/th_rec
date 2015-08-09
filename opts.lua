#!/usr/bin/env th
-- Parse the arguments.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015

local M = {}

function M.parse(arg)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Torch-7 Training script')
  cmd:text()
  cmd:text('Options:')
  cmd:addTime()
  cmd:option('-dbe', 'car', 'database name')
  cmd:option('-ver', 'v1c', 'version')
  cmd:option('-con', 'alex', 'configuration')
  cmd:option('-bufferSize', 1280, 'buffer size')
  cmd:option('-testonly', false, 'Just test loaded net on validation set')
  cmd:option('-threads', 8, 'number of threads')
  cmd:option('-type', 'cuda', 'float or cuda')
  cmd:option('-devid', 1, 'device ID (if using CUDA)')
  cmd:option('-shuffle', true, 'shuffle training samples')
  opt = cmd:parse(arg or {})

  local dbe = opt.dbe
  local ver = opt.ver
  local con = opt.con

  -- folder
  opt.network = string.format('./Models/%s_%s_%s', dbe, ver, con)
  opt.conf = string.format('./Models/%s_%s_conf', dbe, ver)
  opt.saveFold = string.format('./save/%s/torch', dbe)
  opt.dataPath = string.format('data_%s_%s', dbe, ver)

  -- log
  opt.logFold = string.format('%s/log', opt.saveFold)
  os.execute('mkdir -p ' .. opt.logFold)
  opt.logPath = string.format('%s/%s_%s_%s.log', opt.logFold, dbe, ver, con)
  cmd:log(opt.logPath)

  -- output
  opt.modFold = string.format('%s/model', opt.saveFold)
  os.execute('mkdir -p ' .. opt.modFold)
  opt.modPath = string.format('%s/%s_%s_%s', opt.modFold, dbe, ver, con)

  -- cuda
  torch.setnumthreads(opt.threads)
  cutorch.setDevice(opt.devid)
  torch.setdefaulttensortype('torch.FloatTensor')

  return opt
end

return M
