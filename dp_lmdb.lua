#!/usr/bin/env th
-- LMDP Data provider.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09

require 'eladtools'
require 'xlua'
require 'lmdb'
local lib = require('lua_lib')
local Threads = require 'threads'
local ffi = require 'ffi'

-- upvalue set in function init
local sampleSiz, InputSize, meanInfo, trLmdb, teLmdb, DataMean, DataStd, cmp

local provider = {}

----------------------------------------------------------------------
-- Normalize the data by subtracting the mean and dividing the scale.
--
-- Input
--   data  -  b x d x h x w
local function normalize(data)
  local data = data:float()
  for j = 1, 3 do
    data[{{}, j, {}, {}}]:add(-DataMean[j])
    data[{{}, j, {}, {}}]:div(DataStd[j])
  end

  return data
end

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
  for j = 1, 3 do
    data[{{}, j, {}, {}}]:mul(DataStd[j])
    data[{{}, j, {}, {}}]:add(DataMean[j])
  end

  return data
end

----------------------------------------------------------------------
-- Extract from LMDB.
--
-- Input
--   data0   -  original data
--   label0  -  label
local function ExtractSampleFunc(data0, label0)
  local data = data0
  local label = label0

  return normalize(data), label
end

----------------------------------------------------------------------
-- Extract from LMDB training data.
--
-- Input
--   key    -  key
--   data   -  data
--
-- Output
--   img    -  image
--   class  -  class
local function ExtractFromLMDBTrain(key, data)
  -- class
  local class = data.c

  -- decompress
  local img
  if cmp then
    img = image.decompressJPG(data.img)
  else
    img = data.img:float() / 255
  end

  -- random crop
  local nDim = img:dim()
  local start_x = math.random(img:size(nDim) - InputSize)
  local start_y = math.random(img:size(nDim - 1) - InputSize)
  img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

  -- flip
  local hflip = math.random(2) == 1
  if hflip then
    img = image.hflip(img)
  end

  return img:float(), class
end

----------------------------------------------------------------------
-- Extract from LMDB testing data.
--
-- Input
--   key    -  key
--   data   -  data
--
-- Output
--   img    -  image
--   class  -  class
local function ExtractFromLMDBTest(key, data)
  -- class
  local class = data.c

  -- decompress
  local img
  if cmp then
    img = image.decompressJPG(data.img)
  else
    img = data.img:float() / 255
  end

  -- crop the center
  local nDim = img:dim()
  local start_x = math.ceil((img:size(nDim) - InputSize) / 2)
  local start_y = math.ceil((img:size(nDim - 1) - InputSize) / 2)
  img = img:narrow(nDim, start_x, InputSize):narrow(nDim - 1, start_y, InputSize)

  return img:float(), class
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
  local lib = require('lua_lib')
  lib.prIn('data_load init')

  -- compress
  if solConf.cmp == nil then
    cmp = true
  else
    cmp = solConf.cmp
  end

  -- dimension
  sampleSiz = solConf.smpSiz or {3, 224, 224}
  InputSize = sampleSiz[2]

  -- mean
  meanInfo = torch.load(opt.PATH.meanPath)
  DataMean = meanInfo.me
  DataStd = meanInfo.std

  -- train
  trLmdb = opt.PATH.trLmdb
  if not cmp then
    trLmdb = trLmdb .. '_ori'
  end
  local trLmdb2 = trLmdb:gsub(paths.home, '/workplace/feng')
  if paths.dirp(trLmdb2) then
    trLmdb = trLmdb2
    lib.pr('local tr lmdb: %s', trLmdb)
  end

  -- test
  teLmdb = opt.PATH.teLmdb
  if not cmp then
    teLmdb = teLmdb .. '_ori'
  end
  local teLmdb2 = teLmdb:gsub(paths.home, '/workplace/feng')
  if paths.dirp(teLmdb2)  then
    teLmdb = teLmdb2
    lib.pr('local te lmdb: %s', teLmdb)
  end

  -- create threads for dataloader
  local trDB = provider.newTr()
  local teDB = provider.newTe()

  lib.prOut()
  return trDB, teDB
end

----------------------------------------------------------------------
-- Create training data-provider.
--
-- Output
--   db  -  data provider
function provider.newTr()
  local db = eladtools.LMDBProvider {
    Source = lmdb.env({Path = trLmdb, RDONLY = true}),
    SampleSize = sampleSiz,
    ExtractFunction = ExtractFromLMDBTrain,
    Name = 'train'
  }
  db:Threads()
  return db
end

----------------------------------------------------------------------
-- Create testing data-provider.
--
-- Output
--   db  -  data provider
function provider.newTe()
  local db = eladtools.LMDBProvider {
    Source = lmdb.env({Path = teLmdb, RDONLY = true}),
    SampleSize = sampleSiz,
    ExtractFunction = ExtractFromLMDBTest,
    Name = 'test'
  }
  db:Threads()
  return db
end

-- upvalue set in function fordInit
local nImg, batchSiz, bufSiz, bufSts, nBuf, bufSrcs, iBuf, iBufSrc, iBat, nBat
local MiniBatch
local nBufSrc = 2

----------------------------------------------------------------------
-- Move the next buffer.
--
-- Input
--   DB  -  lmdb provider
local function BufferNext(DB)
  iBufSrc = iBufSrc % nBufSrc + 1
  if iBuf > nBuf then
    bufSrcs[iBufSrc] = nil
    return
  end

  local bufSizi = math.min(bufSiz, nImg - bufSts[iBuf] + 1)
  bufSizi = math.floor(bufSizi / batchSiz) * batchSiz

  -- copy from LMDB provider
  local smpSiz = sampleSiz or {3, 224, 224}
  bufSrcs[iBufSrc].Data:resize(bufSizi, unpack(smpSiz))
  bufSrcs[iBufSrc].Labels:resize(bufSizi)
  local key = string.format('%07d', bufSts[iBuf])
  DB:AsyncCacheSeq(key, bufSizi, bufSrcs[iBufSrc].Data, bufSrcs[iBufSrc].Labels)
  iBuf = iBuf + 1
end

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
function provider.fordInit(DB, train, epoch, opt, solConf)
  lib.prIn('fordInit')

  -- dimension
  nImg = DB:size()

  -- info
  batchSiz = solConf.batchSiz
  bufSiz = solConf.bufSiz


  -- TODO: fix this for testing
  nImg = math.floor(nImg / batchSiz) * batchSiz

  -- buffers position
  bufSts = torch.range(1, nImg, bufSiz):long()
  nBuf = bufSts:size(1)
  if train and opt.shuffle then
    bufSts = bufSts:index(1, torch.randperm(nBuf):long())
  end

  -- create buffer src
  bufSrcs = {}
  for i = 1, nBufSrc do
    bufSrcs[i] = DataProvider({Source = {torch.FloatTensor(),
                                         torch.IntTensor()}})
  end

  -- next buffer
  iBuf = 1
  iBufSrc = 1

  -- mini batch
  MiniBatch = DataProvider {
    Name = 'minibatch',
    MaxNumItems = batchSiz,
    Source = bufSrcs[iBufSrc],
    ExtractFunction = ExtractSampleFunc,
    TensorType = 'torch.CudaTensor'
  }

  -- init buffer
  BufferNext(DB)

  -- batch position
  iBat = 1
  nBat = bufSiz / batchSiz
  local nMini = nImg / batchSiz

  lib.pr('nImg %d, batchSiz %d', nImg, batchSiz)
  lib.prOut()
  return nImg, batchSiz, nMini
end

----------------------------------------------------------------------
-- Reset forward.
--
-- Input
--   DB     -  lmdb provider
--   train  -  train or test
--   opt    -  option
function provider.fordReset(DB, train, opt)
  DB:Synchronize()
  MiniBatch:Reset()
  MiniBatch.Source = bufSrcs[iBufSrc]

  if train and opt.shuffle then
    MiniBatch.Source:ShuffleItems()
  end

  -- update buffer asynchronously
  BufferNext(DB)
end

----------------------------------------------------------------------
-- Move to the next batch.
--
-- Input
--   DB     -  lmdb provider
--   train  -  train or test
--   opt    -  option
function provider.fordNextBatch(DB, train, opt)
  -- init
  if iBat == 1 then
    provider.fordReset(DB, train, opt)
  end

  -- get next batch
  local flag = MiniBatch:GetNextBatch()
  assert(flag)

  -- move to next batch
  iBat = iBat + 1
  if iBat > nBat then
    iBat = 1
  end

  return MiniBatch.Data, MiniBatch.Labels
end

return provider
