#!/usr/bin/env th
-- Test using Torch.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-20-2015

require('torch')
require('xlua')
require('optim')
require('pl')
require('eladtools')
require('trepl')
require('net.alex')
local th = require('lua_th')
local lib = require('lua_lib')

-- argument
local opts = require('opts')
opt = opts.parse(arg, 'test')

-- data
local dat = ThDat(opt.dbe, opt.ver)

-- network
local net = ThNet(opt.dbe, opt.ver, opt.con)
local _, _, solConf, _ = dofile(opt.CONF.protTr)
rawset(_G, 'solConf', solConf)

-- data loader
local data_load = require('data_load')

-- trained model
local model = torch.load(net.CONF.modPath)
model:evaluate()

-- each image
local ha = lib.lmdbRIn(dat.PATH.teLmdb)
local co = 0
lib.prCIn('img', ha.n, .1)
for iImg = 1, ha.n do
  lib.prC(iImg)
  -- read one
  local key, val = lib.lmdbR(ha)

  -- extract img
  local img, c = data_load.ExtractFromLMDBTest(key, val)
  local img2 = torch.Tensor(1, unpack(solConf.smpSiz))
  img2[1] = img

  -- normalize
  img = data_load.Normalize(img2)
  img = img:type('torch.CudaTensor')

  -- classify
  local y = model:forward(img)

  -- accuracy
  local a, b = y:max(2)

  if b[1][1] == c then
    co = co + 1
  end
end
lib.prCOut(ha.n)

print(string.format('%d/%d', co, ha.n))

lib.lmdbROut(ha)
