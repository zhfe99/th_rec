#!/usr/bin/env th
-- Mnist data provider.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-08

local lib = require('lua_lib')

-- upvalue set in function init
local sampleSiz, InputSize, meanInfo, trLmdb, teLmdb, DataMean, DataStd, cmp

local provider = {}

----------------------------------------------------------------------
-- De-normalize the data.
--
-- Input
--   data  -  b x d x h x w
--
-- Output
--   data  -  b x d x h x w
function provider.denormalize(data)
  local data = data:float()
  data:mul(DataStd)
  data:add(DataMean)
  return data
end

----------------------------------------------------------------------
-- Distort image.
--
-- Input
--   foo  -  foo, n x d x h x w
--
-- Output
--   res  -  distorted image, n x d x h x w
local function distortData32(foo)
  local hNew = 60
  local res = torch.FloatTensor(foo:size(1), 1, 32, 32):fill(0)

  -- each image
  for i = 1, foo:size(1) do
    -- original
    baseImg = foo:select(1, i)

    -- rotate
    r = image.rotate(baseImg, torch.uniform(-3.14 / 4, 3.14 / 4))

    -- scale
    scale = torch.uniform(0.7, 1.2)
    sz = torch.floor(scale * 32)
    s = image.scale(r, sz, sz)

    -- translate
    rest = hNew - sz
    offsetx = torch.random(1, 1 + rest)
    offsety = torch.random(1, 1 + rest)

    local distImg = torch.FloatTensor(1, hNew, hNew):fill(0)
    distImg:narrow(2, offsety, sz):narrow(3, offsetx, sz):copy(s)

    -- image.save('s.jpg', s)
    -- image.save('distImg.jpg', distImg)
    -- local debugger = require('fb.debugger')
    -- debugger.enter()

    res:select(1, i):copy(image.scale(distImg, 32, 32))
  end
  return res
end

----------------------------------------------------------------------
-- Initialize data-loader.
--
-- Input
--   opt      -  option
--   solConf  -  solver configuration
--
-- Output
--   trDB     -  train DB
--   teDB     -  test DB
function provider.init(opt, solConf)
  lib.prIn('dp_mnist.init')

  -- dimension
  sampleSiz = solConf.smpSiz or {1, 32, 32}
  InputSize = sampleSiz[2]

  -- path
  local trBinPath = paths.concat(paths.home, 'data/mnist/v0/train_32x32.t7')
  local teBinPath = paths.concat(paths.home, 'data/mnist/v0/test_32x32.t7')

  -- load
  local trBin = torch.load(trBinPath, 'ascii')
  local teBin = torch.load(teBinPath, 'ascii')
  trBin.data = trBin.data:float()
  teBin.data = teBin.data:float()
  trBin.labels = trBin.labels:float()
  teBin.labels = teBin.labels:float()

  -- distort
  trBin.data = distortData32(trBin.data)
  teBin.data = distortData32(teBin.data)

  -- normalize
  DataMean = trBin.data:mean()
  DataStd = trBin.data:std()
  trBin.data:add(-DataMean):div(DataStd)
  teBin.data:add(-DataMean):div(DataStd)

  -- create
  -- local trDB = {}
  -- local teDB = {}

  lib.prOut()
  return trBin, teBin
end

-- up-values
local batchSize

----------------------------------------------------------------------
-- Initalize forward passing.
--
-- Input
--   DB       -  lmdb data provider
--   train    -  train or test
--   epoch    -  epoch id
--   opt      -  option
--   solConf  -  solver configuration
--
-- Output
--   nImg     -  #total image
--   batchSiz -  batch size
--   nMini    -  #mini-batch
function provider.fordInit(bin, train, epoch, opt, solConf)
  lib.prIn('dp_mnist.fordInit')
  batchSize = 256

  local nImg = bin.data:size(1)
  local nMini = torch.floor(nImg / batchSize)

  -- store
  bin.iMini = 1
  bin.nMini = nMini

  lib.prOut()
  return nImg, batchSiz, nMini
end

----------------------------------------------------------------------
-- Move to the next batch.
--
-- Input
--   DB     -  lmdb provider
--   train  -  train or test
--   opt    -  option
--
-- Output
--   data   -  data, n x d x h x w
--   labels -  labels, n x
function provider.fordNextBatch(bin, train, opt)
  lib.prIn('dp_mnist.fordNextBatch')

  -- fetch
  local st = (bin.iMini - 1) * batchSize + 1
  local data = bin.data:narrow(1, st, batchSize)
  local labels = bin.labels:narrow(1, st, batchSize)

  -- move to next batch
  bin.iMini = bin.iMini + 1
  if bin.iMini > bin.nMini then
    bin.iMini = 1
  end

  lib.prOut()
  return data:cuda(), labels:cuda()
end

return provider
