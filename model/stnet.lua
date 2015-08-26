#!/usr/bin/env th
-- STN network.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-06-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-26-2015

require 'stn'

local stn = {}


----------------------------------------------------------------------
-- Create a transformation module including:
--   nn.Linear
--   AffineTransformMatrixGenerator
--   AffineGridGeneratorBHWD
-- also
--   initialize nn.Linear
--
-- Input
--   tran    -  transformation, 'aff' | 'sca' | 'tra' | 'rot' | 'tra2' | 'tras'
--   k       -  #dimension of the output of the locnet
--   inSiz   -  input size
--
-- Output
--   traNet  -  transformation net
--   traMod  -  transformation module
function stn.newTran(tran, k, inSiz)
  local traNet = nn.Sequential()

  -- out layer
  local traMod
  if tran == 'aff' then
    -- affine
    traMod = nn.Linear(k, 6)

    -- init
    traMod.weight:fill(0)
    local bias = torch.FloatTensor(6):fill(0)
    bias[1] = 1
    bias[5] = 1
    traMod.bias:copy(bias)

    traNet:add(traMod)
    traNet:add(nn.View(2, 3))

  elseif tran == 'tra' then
    -- translation
    traMod = nn.Linear(k, 2)

    -- init
    traMod.weight:fill(0)
    local bias = torch.FloatTensor(2):fill(0)
    bias[1] = 0
    bias[2] = 0
    traMod.bias:copy(bias)

    traNet:add(traMod)
    traNet:add(nn.AffineTransformMatrixGenerator(false, false, true))

  elseif tran == 'tras2' then
    -- translation + half-scaling
    traMod = nn.Linear(k, 2)

    -- init
    traMod.weight:fill(0)
    local bias = torch.FloatTensor(2):fill(0)
    bias[1] = 0
    bias[2] = 0
    traMod.bias:copy(bias)

    traNet:add(traMod)
    traNet:add(nn.AffineTransformMatrixGenerator(false, false, true, .5))

  elseif tran == 'tras' then
    -- translation + scaling
    traMod = nn.Linear(k, 3)

    -- init
    traMod.weight:fill(0)
    local bias = torch.FloatTensor(3):fill(0)
    bias[1] = 1
    bias[2] = 0
    bias[3] = 0
    traMod.bias:copy(bias)

    traNet:add(traMod)
    traNet:add(nn.AffineTransformMatrixGenerator(false, true, true))

  else
    assert(nil, string.format('unknown tran: %s', tran))
  end

  -- grid generation
  traNet:add(nn.AffineGridGeneratorBHWD(inSiz, inSiz))

  return traNet, traMod
end


----------------------------------------------------------------------
-- Create the stn model.
--
-- Input
--   locnet  -  localization network
--              the output layer should be, nBat x k
--   tran    -  transformation, 'aff' | 'sca' | 'tra' | 'rot' | 'tra2' | 'tras'
--   k       -  #dimension of the output of the locnet
--   inSiz   -  input size
--
-- Output
--   spanet  -  model
--   mods    -  module needed to be updated, m x
function stn.new(locnet, tran, k, inSiz)
  local spanet = nn.Sequential()
  local concat = nn.ConcatTable()

  -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
  local tranet = nn.Sequential()
  tranet:add(nn.Identity())
  tranet:add(nn.Transpose({2, 3}, {3, 4}))

  -- out layer
  local outLayer
  if tran == 'aff' then
    -- affine
    outLayer = nn.Linear(k, 6)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(6):fill(0)
    bias[1] = 1
    bias[5] = 1
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.View(2, 3))

  elseif tran == 'tra' then
    -- translation
    outLayer = nn.Linear(k, 2)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(2):fill(0)
    bias[1] = 0
    bias[2] = 0
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.AffineTransformMatrixGenerator(false, false, true))

  elseif tran == 'tras2' then
    -- translation with fixed half-scaling
    outLayer = nn.Linear(k, 2)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(2):fill(0)
    bias[1] = 0
    bias[2] = 0
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.AffineTransformMatrixGenerator(false, false, true, .5))

  elseif tran == 'tras' then
    -- translation + scaling
    outLayer = nn.Linear(k, 3)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(3):fill(0)
    bias[1] = 1
    bias[2] = 0
    bias[3] = 0
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.AffineTransformMatrixGenerator(false, true, true))

  else
    assert(nil, string.format('unknown tran: %s', tran))
  end

  -- grid generation
  locnet:add(nn.AffineGridGeneratorBHWD(inSiz, inSiz))

  -- we need a table input for the bilinear sampler, so we use concattable
  concat:add(tranet)
  concat:add(locnet)

  spanet:add(concat)
  spanet:add(nn.BilinearSamplerBHWD())

  -- and we transpose back to standard BDHW format for subsequent processing by nn modules
  spanet:add(nn.Transpose({3, 4}, {2, 3}))

  return spanet, {outLayer}
end

----------------------------------------------------------------------
-- Create the stn model.
--
-- Input
--   locNet  -  localization network
--              the output layer should be, nBat x k
--   tran    -  transformation, 'aff' | 'sca' | 'tra' | 'rot' | 'tra2' | 'tras'
--   k       -  #dimension of the output of the locnet
--   inSiz   -  input size
--   m       -  #transformation
--
-- Output
--   spaNet  -  model
--   mods    -  module needed to be updated, m x
function stn.new2(locNet, tran, k, inSiz, m)
  local spaNet = nn.Sequential()
  local concat = nn.ConcatTable()
  spaNet:add(concat)

  -- transpose inputs to BHWD, for the bilinear sampler
  local ideNet = nn.Sequential()
  concat:add(ideNet)
  ideNet:add(nn.Identity())
  ideNet:add(nn.Transpose({2, 3}, {3, 4}))

  -- transformation layer
  concat:add(locNet)
  local tranNets = nn.ConcatTable()
  locNet:add(tranNets)
  local tranMods = {}
  for i = 1, m do
    local tranNet, tranMod = stn.newTran(tran, k, inSiz)
    tranNets:add(tranNet)
    table.insert(tranMods, tranMod)
  end

  -- sampler
  local smpNets = nn.ConcatTable()
  spaNet:add(smpNets)
  for i = 1, m do
    local smpNet = nn.Sequential()
    smpNets:add(smpNet)

    local selNet = nn.ConcatTable()
    smpNet:add(selNet)

    -- select the transposed input
    selNet:add(nn.SelectTable(1))

    -- select the grid
    local tmpNet = nn.Sequential()
    selNet:add(tmpNet)
    tmpNet:add(nn.SelectTable(2)):add(nn.SelectTable(i))

    -- sample
    smpNet:add(nn.BilinearSamplerBHWD())

    -- transpose back to standard BDHW format
    smpNet:add(nn.Transpose({3, 4}, {2, 3}))
  end

  return spaNet, tranMods
end

return stn
